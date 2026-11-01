"""
04_evaluate.py -- valutazione di UN run: discriminazione + calibrazione (interna ed esterna).

    python 04_evaluate.py --run <run_id>

1. Logit su NIH val, NIH test, VinDr test con il preprocessing NATIVO del run
   (risolve il limite "condizioni senza crop valutate con il crop").
2. Ricalibrazione post-hoc per classe stimata SOLO su NIH val:
       raw | temperature scaling (T) | Platt scaling (a, b)
   e applicata invariata a NIH test e VinDr test.
3. Metriche: ROC-AUC, PR-AUC, ECE (10 bin uniformi, pesato -- formula standard),
   ECE adattivo, Brier, NLL, slope/intercept di calibrazione; soglie di Youden
   da val -> sens/spec/PPV/F1 su test.

Output in runs/<run_id>/: preds_{nih_val,nih_test,vindr_test}.npz, eval_perclass.csv, eval.json
Su VinDr il target "Nodule/Mass" usa max(logit Mass, logit Nodule); la sua calibrazione
e' riportata ma va interpretata con cautela (unione di due classi NIH).
"""
import os
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from common import (load_config, run_dir, load_manifest, CachedDataset, build_transforms, make_loader,
                    get_device, load_model_for_run, predict_logits, CLASSES, CIDX, VINDR_TARGETS,
                    sigmoid_np, calib_metrics, fit_temperature, fit_platt, youden_threshold, dump_json)
from experiments import find_run


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def safe_ap(y, s):
    return float(average_precision_score(y, s)) if y.sum() > 0 else np.nan


def get_preds(cfg, run, model, device, name, tf):
    rd = run_dir(cfg, run["run_id"])
    p = os.path.join(rd, f"preds_{name}.npz")
    df = load_manifest(cfg, name)
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True)
        if len(d["image_id"]) == len(df):
            return df, d["logits"]
    loader = make_loader(CachedDataset(df, tf), cfg["train"]["batch_size"] * 2, False, cfg["train"]["num_workers"])
    z = predict_logits(model, loader, device, cfg)
    np.savez_compressed(p, logits=z.astype(np.float32), image_id=df["image_id"].values.astype(str))
    return df, z


def recal(z, method, params):
    if method == "raw":
        return sigmoid_np(z)
    if method == "ts":
        return sigmoid_np(z / params)
    a, b = params
    return sigmoid_np(a * z + b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.smoke)
    run = find_run(cfg, a.run)
    rd = run_dir(cfg, run["run_id"])
    if os.path.exists(os.path.join(rd, "eval.json")):
        print(f"[SKIP] {run['run_id']} gia' valutato")
        return
    device = get_device()
    model = load_model_for_run(cfg, run, device)
    _, eval_tf = build_transforms(cfg["train"], run["crop"], False)

    va_df, z_va = get_preds(cfg, run, model, device, "nih_val", eval_tf)
    te_df, z_te = get_preds(cfg, run, model, device, "nih_test", eval_tf)
    y_va, y_te = va_df[CLASSES].values.astype(float), te_df[CLASSES].values.astype(float)
    has_vd = True
    try:
        vd_df, z_vd = get_preds(cfg, run, model, device, "vindr_test", eval_tf)
    except FileNotFoundError:
        has_vd = False

    rows = []
    # ---- NIH: 14 classi ------------------------------------------------
    for c in CLASSES:
        i = CIDX[c]
        zv, yv, zt, yt = z_va[:, i], y_va[:, i], z_te[:, i], y_te[:, i]
        if yv.sum() == 0:
            continue
        T = fit_temperature(zv, yv)
        ab = fit_platt(zv, yv)
        thr = youden_threshold(sigmoid_np(zv), yv)
        pt = sigmoid_np(zt)
        pred = pt >= thr
        tp, fp = float((pred & (yt == 1)).sum()), float((pred & (yt == 0)).sum())
        fn, tn = float((~pred & (yt == 1)).sum()), float((~pred & (yt == 0)).sum())
        base = dict(dataset="nih_test", target=c, n=len(yt), n_pos=int(yt.sum()),
                    auc=safe_auc(yt, zt), pr_auc=safe_ap(yt, zt), T=T, platt_a=ab[0], platt_b=ab[1],
                    youden_thr=thr, sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                    ppv=tp / max(tp + fp, 1), f1=2 * tp / max(2 * tp + fp + fn, 1))
        for m, prm in [("raw", None), ("ts", T), ("platt", ab)]:
            for k, v in calib_metrics(recal(zt, m, prm), yt).items():
                base[f"{m}_{k}"] = v
        rows.append(base)
    params = {r["target"]: (r["T"], (r["platt_a"], r["platt_b"])) for r in rows}

    # ---- VinDr: target mappati, parametri di ricalibrazione da NIH val ----
    if has_vd:
        for tgt, (_, nih_cls) in VINDR_TARGETS.items():
            if f"vd_{tgt}" not in vd_df.columns:
                continue   # classe non disponibile in questa sorgente VinDr
            idx = [CIDX[c] for c in nih_cls]
            zt = z_vd[:, idx].max(1)
            yt = vd_df[f"vd_{tgt}"].values.astype(float)
            base = dict(dataset="vindr_test", target=tgt, n=len(yt), n_pos=int(yt.sum()),
                        auc=safe_auc(yt, zt), pr_auc=safe_ap(yt, zt), merged=len(idx) > 1)
            if len(idx) == 1 and nih_cls[0] in params and yt.sum() > 0:
                T, ab = params[nih_cls[0]]
                for m, prm in [("raw", None), ("ts", T), ("platt", ab)]:
                    for k, v in calib_metrics(recal(zt, m, prm), yt).items():
                        base[f"{m}_{k}"] = v
            rows.append(base)

    pc = pd.DataFrame(rows)
    for k, v in run.items():
        pc[k] = v
    pc.to_csv(os.path.join(rd, "eval_perclass.csv"), index=False)

    summ = dict(run)
    for ds, g in pc.groupby("dataset"):
        g_cal = g[~(g["merged"] == True)] if "merged" in g.columns else g  # noqa: E712
        summ[f"{ds}_macro_auc"] = float(g["auc"].mean())
        summ[f"{ds}_macro_prauc"] = float(g["pr_auc"].mean())
        for m in ["raw", "ts", "platt"]:
            for k in ["ece", "ece_adaptive", "brier", "nll", "slope", "intercept"]:
                col = f"{m}_{k}"
                if col in g_cal:
                    summ[f"{ds}_{col}"] = float(g_cal[col].mean())
    # micro AUC NIH (continuita' con il paper)
    summ["nih_test_micro_auc"] = float(roc_auc_score(y_te.ravel(), z_te.ravel()))
    dump_json(summ, os.path.join(rd, "eval.json"))
    print(f"[DONE] eval {run['run_id']}: NIH AUC={summ.get('nih_test_macro_auc'):.4f} "
          f"VinDr AUC={summ.get('vindr_test_macro_auc', float('nan')):.4f} "
          f"NIH ECE raw={summ.get('nih_test_raw_ece'):.4f} ts={summ.get('nih_test_ts_ece'):.4f}")


if __name__ == "__main__":
    main()
