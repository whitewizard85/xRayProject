"""
05b_attribution_ext.py -- attribuzione ESTESA per UN run (revisione 2026-11).

    python 05b_attribution_ext.py --run <run_id>

Risponde alle due obiezioni piu' prevedibili al test di sensibilita' del paper:

1. RANDOMIZZAZIONE A CASCATA COMPLETA (Adebayo et al., 2018). Il paper re-inizializzava solo
   l'ultimo stadio e la testa. Qui i livelli del modello vengono re-inizializzati in modo
   CUMULATIVO dall'alto verso il basso (testa, ultimo stadio, ..., stem) e dopo ogni livello si
   misura la correlazione di Spearman fra la mappa originale e quella del modello randomizzato.
   Re-inizializzazione uniforme per tutte le architetture (common.reinit_modules).

2. SECONDA BASELINE PER INTEGRATED GRADIENTS. Oltre alla baseline nera del paper, IG con baseline
   = copia sfocata dell'immagine (sigma configurabile). Per IG-blur si calcolano localizzazione,
   faithfulness (deletion) e randomizzazione a cascata sugli STESSI sottoinsiemi fissi del paper.

Coppie: le stesse di 05_attribution.py (build_pairs), quindi confronti appaiati con il paper.
Cascata: coppie del sottoinsieme "sanity" (solo VinDr, come nel paper). IG-blur: sottoinsieme IG.

Quali run: attribution_ext.runs = "reference" (cella di riferimento di tutti i blocchi: 50 run) oppure "all".
Per i run non selezionati scrive solo il marker attr_ext_done.json con skipped=true, cosi' la coda prosegue.

Output in runs/<run_id>/:
  attr_ext_cascade.csv      pair_id, method, level, level_name, spearman
  attr_ext_igblur_pairs.csv localizzazione IG-blur (stesse colonne di attr_pairs.csv)
  attr_ext_igblur_faith.csv deletion test IG-blur
  attr_ext_maps.npz         mappe IG-blur sulla griglia comune (per l'accordo tra seed)
  attr_ext_done.json
"""
import os
import copy
import argparse
import importlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from tqdm import tqdm
from torchvision.transforms.functional import gaussian_blur

from common import (load_config, run_dir, build_transforms, get_device, load_model_for_run, target_layer,
                    feature_layout, cascade_levels, reinit_modules, box_cache_to_input, boxes_to_mask,
                    to_common_grid, IMAGENET_MEAN, IMAGENET_STD, dump_json)
from experiments import find_run

A = importlib.import_module("05_attribution")   # riuso: build_pairs, CamExtractor, IG, smooth, metriche

METHODS_CAM = ["gradcam", "gradcampp"]


def selected(cfg, run):
    mode = cfg.get("attribution_ext", {}).get("runs", "reference")
    if mode == "all":
        return True
    if run["block"] == "regime":
        return True
    return run["crop"] == 0 and run["schedule"] == "fx10" and run["aug"] == 1


def blur_base(x, sigma):
    k = int(2 * round(3 * sigma) + 1)
    return gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])


def maps_for(model, ext, pb, x, a, ae, bases, black):
    """Mappe smussate per un batch: Grad-CAM, Grad-CAM++ e IG per ciascuna baseline richiesta."""
    cams, probs = ext(x, [p["idx"] for p in pb])
    out = {k: A.smooth(v, a["smooth_sigma_px"]) for k, v in cams.items() if k in METHODS_CAM}
    ext.enabled = False
    for bname in bases:
        ig = torch.empty_like(out["gradcam"])
        for b, p in enumerate(pb):
            base = black if bname == "black" else blur_base(x[b:b + 1], ae["blur_sigma_px"])
            ig[b] = A.integrated_gradients(model, x[b:b + 1], p["idx"], a["ig_steps"], a["ig_batch"], base)
        out[f"ig_{bname}"] = A.smooth(ig, a["smooth_sigma_px"])
    ext.enabled = True
    return out, probs


def to_grid_flat(m, G):
    return F.interpolate(m[None, None], size=(G, G), mode="area").flatten().float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.smoke)
    run = find_run(cfg, args.run)
    rd = run_dir(cfg, run["run_id"])
    done = os.path.join(rd, "attr_ext_done.json")
    if os.path.exists(done):
        print(f"[SKIP] {run['run_id']} attribuzione estesa gia' fatta")
        return
    if not selected(cfg, run):
        dump_json(dict(run_id=run["run_id"], skipped=True), done)
        print(f"[SKIP] {run['run_id']} non selezionato (attribution_ext.runs)")
        return
    a, tc, ae = cfg["attribution"], cfg["train"], cfg["attribution_ext"]
    bases = list(ae["ig_baselines"])
    device = get_device()
    torch.backends.cudnn.benchmark = True
    model = load_model_for_run(cfg, run, device).float()
    for p in model.parameters():
        p.requires_grad_(True)
    _, tf = build_transforms(tc, run["crop"], False)
    S, G = tc["image_size"], a["map_grid"]
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    black = ((0.0 - mean) / std).expand(1, 3, S, S)

    pairs = A.build_pairs(cfg)
    casc_idx = [i for i, p in enumerate(pairs) if p["in_sanity"] and p["dataset"] == "vindr"]
    ig_idx = [i for i, p in enumerate(pairs) if p["in_ig"]]
    print(f"[INFO] cascata su {len(casc_idx)} coppie VinDr; IG-blur su {len(ig_idx)} coppie")

    def load_x(idx):
        return torch.stack([tf(Image.open(pairs[i]["cache_path"]).convert("RGB")) for i in idx]).to(device)

    # ---------------- 1) IG con baseline sfocata: localizzazione, faithfulness, mappe -------------
    rows, frows = [], []
    grid = np.full((len(pairs), G, G), np.nan, dtype=np.float16)
    rng = np.random.default_rng(12345)
    bs = a["cam_batch"]
    if "blur" in bases:
        for s in tqdm(range(0, len(ig_idx), bs), desc=f"igblur {run['run_id']}", mininterval=30):
            idx = ig_idx[s:s + bs]
            x = load_x(idx)
            ig = torch.empty((len(idx), S, S), device=device)
            for b, i in enumerate(idx):
                ig[b] = A.integrated_gradients(model, x[b:b + 1], pairs[i]["idx"], a["ig_steps"], a["ig_batch"],
                                               blur_base(x[b:b + 1], ae["blur_sigma_px"]))
            ig = A.smooth(ig, a["smooth_sigma_px"])
            with torch.no_grad():
                probs = torch.sigmoid(model(x)).float()
            grid[idx] = to_common_grid(ig, tc, run["crop"], G).cpu().numpy().astype(np.float16)
            for b, i in enumerate(idx):
                p = pairs[i]
                boxes = [bx for bx in (box_cache_to_input(bx, cfg["data"]["cache_size"], tc, run["crop"])
                                       for bx in p["boxes"]) if bx]
                mask = boxes_to_mask(boxes, S) if boxes else None
                m = ig[b]
                r = dict(run_id=run["run_id"], pair_id=p["pair_id"], dataset=p["dataset"], target=p["target"],
                         method="ig_blur", prob=float(probs[b, p["idx"]].max()), box_visible=mask is not None)
                r.update(A.loc_metrics(m.cpu().numpy(), mask) if mask is not None else
                         dict(energy=np.nan, norm_energy=np.nan, pointing=np.nan, area_frac=0.0))
                rows.append(r)
                if p["in_faith"]:
                    am, ar, p0 = A.deletion_curve(model, x[b], p["idx"], m, a["faithfulness_fracs"], rng)
                    frows.append(dict(run_id=run["run_id"], pair_id=p["pair_id"], dataset=p["dataset"],
                                      method="ig_blur", del_auc_map=am, del_auc_random=ar, p0=p0, faith_gain=ar - am))

    # ---------------- 2) randomizzazione a cascata completa -------------------------------------
    crow = []
    if casc_idx:
        ext = A.CamExtractor(model, target_layer(model), feature_layout(model))
        orig = {}
        for s in range(0, len(casc_idx), bs):
            idx = casc_idx[s:s + bs]
            x = load_x(idx)
            mp, _ = maps_for(model, ext, [pairs[i] for i in idx], x, a, ae, bases, black)
            for b, i in enumerate(idx):
                orig[i] = {k: to_grid_flat(v[b], G) for k, v in mp.items()}
        ext.remove()

        rmodel = copy.deepcopy(model)
        levels = cascade_levels(rmodel)
        for li, (lname, mods) in enumerate(tqdm(levels, desc=f"cascade {run['run_id']}")):
            reinit_modules(mods, seed=ae["reinit_seed"] + li)       # cumulativo: i livelli precedenti restano random
            rext = A.CamExtractor(rmodel, target_layer(rmodel), feature_layout(rmodel))
            for s in range(0, len(casc_idx), bs):
                idx = casc_idx[s:s + bs]
                x = load_x(idx)
                mp, _ = maps_for(rmodel, rext, [pairs[i] for i in idx], x, a, ae, bases, black)
                for b, i in enumerate(idx):
                    for meth, v in mp.items():
                        o, r = orig[i][meth], to_grid_flat(v[b], G)
                        ok = np.isfinite(o).all() and np.isfinite(r).all() and o.std() > 0 and r.std() > 0
                        rho = spearmanr(o, r).statistic if ok else np.nan
                        crow.append(dict(run_id=run["run_id"], pair_id=pairs[i]["pair_id"], method=meth,
                                         level=li + 1, level_name=lname, n_levels=len(levels),
                                         spearman=float(rho), constant_map=bool(r.std() == 0)))
            rext.remove()

    pd.DataFrame(rows).to_csv(os.path.join(rd, "attr_ext_igblur_pairs.csv"), index=False)
    pd.DataFrame(frows).to_csv(os.path.join(rd, "attr_ext_igblur_faith.csv"), index=False)
    pd.DataFrame(crow).to_csv(os.path.join(rd, "attr_ext_cascade.csv"), index=False)
    np.savez_compressed(os.path.join(rd, "attr_ext_maps.npz"), pair_id=np.array([p["pair_id"] for p in pairs]),
                        map_ig_blur=grid)
    summ = dict(run_id=run["run_id"], skipped=False, n_cascade_pairs=len(casc_idx), n_igblur_pairs=len(ig_idx))
    if crow:
        c = pd.DataFrame(crow)
        last = c[c["level"] == c["level"].max()]
        for meth, v in last.groupby("method")["spearman"].mean().items():
            summ[f"cascade_full_{meth}_rho"] = float(v)
    if rows:
        r = pd.DataFrame(rows)
        for (ds, m), g in r.groupby(["dataset", "method"]):
            summ[f"{ds}_{m}_energy"] = float(g["energy"].mean())
    dump_json(summ, done)
    print(f"[DONE] attribuzione estesa {run['run_id']}: {summ}")


if __name__ == "__main__":
    main()
