"""
03_train.py -- allena UN run del piano (ripristinabile).

    python 03_train.py --run <run_id>

- Salva last.pt a fine di OGNI epoca (modello+ottimizzatore+scheduler+stato):
  se il server si spegne, rilanciando riparte dall'epoca successiva.
- L'ordine di shuffle di ogni epoca dipende solo da (seed, epoca), quindi una
  ripresa produce lo stesso ordine di dati di un run ininterrotto.
- best.pt = pesi con la migliore macro ROC-AUC su validation NIH.
- train_done.json = storia delle epoche + tempi; marca il run come completato.
"""
import os
import time
import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from common import (load_config, run_dir, load_manifest, CachedDataset, build_transforms, make_loader,
                    build_model, build_loss, set_seed, get_device, autocast_ctx, predict_logits,
                    CLASSES, dump_json)
from experiments import find_run


def macro_auc(z, y):
    aucs = [roc_auc_score(y[:, i], z[:, i]) for i in range(y.shape[1]) if 0 < y[:, i].sum() < len(y)]
    return float(np.mean(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.smoke)
    run = find_run(cfg, a.run)
    rd = run_dir(cfg, run["run_id"])
    done_flag = os.path.join(rd, "train_done.json")
    if os.path.exists(done_flag):
        print(f"[SKIP] {run['run_id']} gia' allenato")
        return
    dump_json(run, os.path.join(rd, "run.json"))
    tc = cfg["train"]
    device = get_device()
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tr_df, va_df = load_manifest(cfg, "nih_train"), load_manifest(cfg, "nih_val")
    train_tf, eval_tf = build_transforms(tc, run["crop"], run["aug"])
    tr_ds = CachedDataset(tr_df, train_tf, CLASSES)
    va_ds = CachedDataset(va_df, eval_tf, CLASSES)
    va_loader = make_loader(va_ds, tc["batch_size"] * 2, False, tc["num_workers"])
    y_val = va_ds.labels

    set_seed(run["seed"])
    model = build_model(run["arch"], pretrained=tc["pretrained"], override=tc["model_override"]).to(device)
    if tc["channels_last"] and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    pos = tr_ds.labels.sum(0)
    pos_weight = torch.tensor((len(tr_ds) - pos) / np.maximum(pos, 1), dtype=torch.float32, device=device)
    criterion = build_loss(run["loss"], pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])

    if run["schedule"] == "fx10":
        epochs, patience = tc["fixed_epochs"], None
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    elif run["schedule"] == "es20":
        epochs, patience = tc["max_epochs_cap"], tc["patience"]
        sched = None
    else:
        raise ValueError(run["schedule"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and tc["amp"] == "fp16"))

    last_p, best_p = os.path.join(rd, "last.pt"), os.path.join(rd, "best.pt")
    start_epoch, best_auc, no_improve, history = 1, -1.0, 0, []
    if os.path.exists(last_p):
        st = torch.load(last_p, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if sched is not None and st["sched"] is not None:
            sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, best_auc, no_improve, history = st["epoch"] + 1, st["best_auc"], st["no_improve"], st["history"]
        torch.set_rng_state(st["rng_cpu"])
        if device.type == "cuda" and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(st["rng_cuda"])
        print(f"[RESUME] {run['run_id']} da epoca {start_epoch} (best val AUC {best_auc:.4f})")

    stopped_early = bool(history and history[-1].get("stop"))
    for epoch in range(start_epoch, epochs + 1):
        if stopped_early:
            break
        t0 = time.time()
        model.train()
        loader = make_loader(tr_ds, tc["batch_size"], True, tc["num_workers"], seed=run["seed"] * 1000 + epoch)
        losses = []
        for x, y, _ in tqdm(loader, desc=f"{run['run_id']} ep{epoch}/{epochs}", mininterval=30):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if tc["channels_last"] and device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(cfg, device):
                out = model(x)
            loss = criterion(out.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss non finita all'epoca {epoch}")
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        if sched is not None:
            sched.step()
        z_val = predict_logits(model, va_loader, device, cfg)
        auc = macro_auc(z_val, y_val)
        improved = auc > best_auc
        if improved:
            best_auc, no_improve = auc, 0
            torch.save(model.state_dict(), best_p + ".tmp")
            os.replace(best_p + ".tmp", best_p)
        else:
            no_improve += 1
        stop = patience is not None and no_improve >= patience
        rec = dict(epoch=epoch, train_loss=float(np.mean(losses)), val_macro_auc=auc, best=improved,
                   seconds=time.time() - t0, imgs_per_s=len(tr_ds) / max(time.time() - t0, 1e-6), stop=stop)
        history.append(rec)
        print(f"[EPOCH] {rec}")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict() if sched is not None else None, scaler=scaler.state_dict(),
                        epoch=epoch, best_auc=best_auc, no_improve=no_improve, history=history,
                        rng_cpu=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None),
                   last_p + ".tmp")
        os.replace(last_p + ".tmp", last_p)
        if stop:
            print(f"[STOP] early stopping all'epoca {epoch}")
            break

    dump_json(dict(run=run, best_val_macro_auc=best_auc, epochs_run=len(history), history=history,
                   total_hours=sum(h["seconds"] for h in history) / 3600,
                   gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"), done_flag)
    if not tc["keep_resume_checkpoint"] and os.path.exists(last_p):
        os.remove(last_p)
    print(f"[DONE] train {run['run_id']} best val AUC={best_auc:.4f}")


if __name__ == "__main__":
    main()
