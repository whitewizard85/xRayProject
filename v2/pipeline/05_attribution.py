"""
05_attribution.py -- attribuzione, localizzazione, faithfulness e sanity check per UN run.

    python 05_attribution.py --run <run_id>

COPPIE immagine-target (identiche per tutti i run, cosi' i confronti sono appaiati):
  - NIH:   box ufficiali (BBox_List_2017) ristretti al test split
  - VinDr: box di consenso del test set (molte piu' immagini: risolve il limite "131 immagini")
  Piu' box della stessa classe sulla stessa immagine -> unione in un'unica maschera.
  Il target di attribuzione e' SEMPRE la classe ground-truth (risolve l'incoerenza
  top-1 vs ground-truth del paper). Per "Nodule/Mass" il punteggio e' logit(Mass)+logit(Nodule).

METODI: Grad-CAM, Grad-CAM++ (implementati qui, senza normalizzazione min-max), Integrated
Gradients (baseline = immagine nera, punti di Riemann medi). Le mappe sono non negative e
smussate con una gaussiana (sigma configurabile) prima delle metriche.

METRICHE per coppia: pointing game, energy-in-box, energy normalizzata per l'area del box,
probabilita' del target. Su sottoinsiemi FISSI e annidati:
  - faithfulness (deletion): si sostituiscono i pixel piu' attribuiti con una versione sfocata
    e si misura il calo di probabilita'; confronto con un ordinamento casuale;
  - sanity check (Adebayo et al.): si re-inizializzano ultimo stadio + testa e si misura la
    correlazione di Spearman tra mappe originali e "randomizzate" (alta = metodo poco sensibile al modello).
Le mappe vengono salvate su una griglia comune (la regione vista da TUTTE le condizioni)
per l'analisi di accordo tra seed e tra condizioni in 06_analyze.py.
"""
import os
import copy
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from tqdm import tqdm
from torchvision.transforms.functional import gaussian_blur

from common import (load_config, run_dir, shared_data_dir, build_transforms, get_device, load_model_for_run,
                    target_layer, feature_layout, last_block_and_head, randomize_modules, box_cache_to_input, boxes_to_mask,
                    to_common_grid, CIDX, VINDR_TARGETS, IMAGENET_MEAN, IMAGENET_STD, dump_json)
from experiments import find_run


# ---------------------------------------------------------------------
def build_pairs(cfg):
    dd = shared_data_dir(cfg)
    pairs = []
    for ds, man, bbf in [("nih", "nih_test", "nih_bbox_test"), ("vindr", "vindr_test", "vindr_bbox_test")]:
        pb, pm = os.path.join(dd, f"{bbf}.csv"), os.path.join(dd, f"{man}.csv")
        if not (os.path.exists(pb) and os.path.exists(pm)):
            continue
        bb, m = pd.read_csv(pb), pd.read_csv(pm)
        paths = dict(zip(m["image_id"].astype(str), m["cache_path"]))
        bb["image_id"] = bb["image_id"].astype(str)
        bb = bb[bb["image_id"].isin(paths)]
        for (img, tgt), g in sorted(bb.groupby(["image_id", "target"])):
            idx = [CIDX[tgt]] if ds == "nih" else [CIDX[c] for c in VINDR_TARGETS[tgt][1]]
            pairs.append(dict(pair_id=f"{ds}|{img}|{tgt}", dataset=ds, image_id=img, target=tgt,
                              cache_path=paths[img], idx=idx,
                              boxes=g[["x1", "y1", "x2", "y2"]].values.tolist()))
    cap = cfg["attribution"].get("max_pairs_per_dataset")
    if cap:
        capped = []
        for ds in ["nih", "vindr"]:
            sub = [p for p in pairs if p["dataset"] == ds]
            if len(sub) > cap:
                sel = sorted(np.random.default_rng(cfg["attribution"]["subset_seed"]).choice(len(sub), cap, replace=False))
                sub = [sub[i] for i in sel]
            capped += sub
        pairs = capped
    if cfg["_smoke"]:
        keep = []
        for ds in ["nih", "vindr"]:
            keep += [p for p in pairs if p["dataset"] == ds][:12]
        pairs = keep
    a = cfg["attribution"]
    rng = np.random.default_rng(a["subset_seed"])
    order = rng.permutation(len(pairs))
    ig = set(order[:a["ig_max_pairs"]].tolist())
    faith = set(order[:min(a["faithfulness_pairs"], a["ig_max_pairs"])].tolist())
    sanity = set(order[:min(a["sanity_pairs"], a["faithfulness_pairs"], a["ig_max_pairs"])].tolist())
    for i, p in enumerate(pairs):
        p["in_ig"], p["in_faith"], p["in_sanity"] = i in ig, i in faith, i in sanity
    return pairs


# ---------------------------------------------------------------------
class CamExtractor:
    def __init__(self, model, layer, layout="nchw"):
        self.model, self.acts, self.grads, self.enabled = model, None, None, True
        self.layout = layout                     # "nhwc" per Swin (timm)
        self.h = layer.register_forward_hook(self._fh)

    def _fh(self, m, i, o):
        if self.enabled and o.requires_grad:
            self.acts = o
            o.register_hook(self._gh)

    def _gh(self, g):
        self.grads = g

    def __call__(self, x, idx_lists):
        self.enabled = True
        x = x.clone().requires_grad_(False)
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        score = torch.stack([out[b, idx].sum() for b, idx in enumerate(idx_lists)])
        score.sum().backward()
        A, G = self.acts.detach().float(), self.grads.detach().float()
        if self.layout == "nhwc":
            A, G = A.permute(0, 3, 1, 2), G.permute(0, 3, 1, 2)
        gc = F.relu((G.mean((2, 3), keepdim=True) * A).sum(1))
        g2, g3 = G ** 2, G ** 3
        s = A.sum((2, 3), keepdim=True)
        aij = g2 / (2 * g2 + s * g3 + 1e-7)
        aij = torch.where(G != 0, aij, torch.zeros_like(aij))
        w = (F.relu(G) * aij).sum((2, 3), keepdim=True)
        gpp = F.relu((w * A).sum(1))
        S = x.shape[-1]
        up = lambda c: F.interpolate(c.unsqueeze(1), size=(S, S), mode="bilinear", align_corners=False)[:, 0]
        probs = torch.stack([torch.sigmoid(out[b, idx]).max() for b, idx in enumerate(idx_lists)]).detach()
        return {"gradcam": up(gc), "gradcampp": up(gpp)}, probs

    def remove(self):
        self.h.remove()


def integrated_gradients(model, x, idx, steps, batch, base):
    """x: (1,3,S,S). Ritorna |sum_c (x-base) * mean_grad| di forma (S,S)."""
    alphas = (torch.arange(steps, device=x.device, dtype=torch.float32) + 0.5) / steps
    total = torch.zeros_like(x)
    for k in range(0, steps, batch):
        al = alphas[k:k + batch].view(-1, 1, 1, 1)
        xs = (base + al * (x - base)).detach().requires_grad_(True)
        out = model(xs)
        g, = torch.autograd.grad(out[:, idx].sum(), xs)
        total += g.sum(0, keepdim=True)
    attr = ((x - base) * total / steps).sum(1)[0]
    return attr.abs().detach()


def smooth(maps, sigma):
    if sigma <= 0:
        return maps
    k = int(2 * round(3 * sigma) + 1)
    return gaussian_blur(maps.unsqueeze(1), kernel_size=[k, k], sigma=[sigma, sigma])[:, 0]


def loc_metrics(m, mask):
    tot = float(m.sum())
    area = float(mask.mean())
    if tot <= 0 or area <= 0:
        return dict(energy=np.nan, norm_energy=np.nan, pointing=np.nan, area_frac=area)
    e = float((m * mask).sum() / tot)
    am = int(np.argmax(m))
    return dict(energy=e, norm_energy=e / area, pointing=float(mask.flat[am]), area_frac=area)


@torch.no_grad()
def deletion_curve(model, x, idx, m, fracs, rng):
    """x: (3,S,S); m: (S,S) mappa. Ritorna (auc_map, auc_random, p0)."""
    S = x.shape[-1]
    xb = gaussian_blur(x.unsqueeze(0), kernel_size=[51, 51], sigma=[16.0, 16.0])[0]
    flat = m.flatten() + torch.rand_like(m.flatten()) * 1e-9 * (m.max() + 1e-12)
    order_map = torch.argsort(flat, descending=True)
    order_rnd = torch.from_numpy(rng.permutation(S * S)).to(x.device)
    batch = []
    for order in (order_map, order_rnd):
        for f in fracs:
            k = int(round(f * S * S))
            M = torch.zeros(S * S, device=x.device)
            M[order[:k]] = 1.0
            M = M.view(1, S, S)
            batch.append(x * (1 - M) + xb * M)
    out = torch.sigmoid(model(torch.stack(batch)))[:, idx].max(1).values.float().cpu().numpy()
    n = len(fracs)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(out[:n], fracs)), float(trapz(out[n:], fracs)), float(out[0])


# ---------------------------------------------------------------------
def compute_maps(model, pairs_batch, x, cfg, ext, with_ig, base):
    a = cfg["attribution"]
    maps, probs = ext(x, [p["idx"] for p in pairs_batch])
    maps = {k: smooth(v, a["smooth_sigma_px"]) for k, v in maps.items()}
    if "ig" in a["methods"]:
        ig = torch.full_like(maps["gradcam"], float("nan"))
        ext.enabled = False
        for b, p in enumerate(pairs_batch):
            if with_ig[b]:
                ig[b] = integrated_gradients(model, x[b:b + 1], p["idx"], a["ig_steps"], a["ig_batch"], base)
        ext.enabled = True
        has = ~torch.isnan(ig[:, 0, 0])
        if has.any():
            ig[has] = smooth(ig[has], a["smooth_sigma_px"])
        maps["ig"] = ig
    return {k: v for k, v in maps.items() if k in a["methods"]}, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.smoke)
    run = find_run(cfg, args.run)
    rd = run_dir(cfg, run["run_id"])
    if os.path.exists(os.path.join(rd, "attr_done.json")):
        print(f"[SKIP] {run['run_id']} attribuzione gia' fatta")
        return
    a, tc = cfg["attribution"], cfg["train"]
    device = get_device()
    torch.backends.cudnn.benchmark = True
    model = load_model_for_run(cfg, run, device).float()
    for p in model.parameters():
        p.requires_grad_(True)
    _, tf = build_transforms(tc, run["crop"], False)
    S, G = tc["image_size"], a["map_grid"]
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    base = ((0.0 - mean) / std).expand(1, 3, S, S)

    pairs = build_pairs(cfg)
    if not pairs:
        print("[WARN] nessuna coppia con box disponibile")
    print(f"[INFO] coppie: {len(pairs)} (IG su {sum(p['in_ig'] for p in pairs)}, "
          f"faithfulness su {sum(p['in_faith'] for p in pairs)}, sanity su {sum(p['in_sanity'] for p in pairs)})")
    ext = CamExtractor(model, target_layer(model), feature_layout(model))
    grid = {m: np.full((len(pairs), G, G), np.nan, dtype=np.float16) for m in a["methods"]}
    rows, frows = [], []
    rng = np.random.default_rng(12345)
    fracs = a["faithfulness_fracs"]
    sanity_maps = {}

    for s in tqdm(range(0, len(pairs), a["cam_batch"]), desc=f"attr {run['run_id']}", mininterval=30):
        pb = pairs[s:s + a["cam_batch"]]
        x = torch.stack([tf(Image.open(p["cache_path"]).convert("RGB")) for p in pb]).to(device)
        maps, probs = compute_maps(model, pb, x, cfg, ext, [p["in_ig"] for p in pb], base)
        for meth, mm in maps.items():
            valid = ~torch.isnan(mm[:, 0, 0])
            gm = torch.full((len(pb), G, G), float("nan"), device=device)
            if valid.any():
                gm[valid] = to_common_grid(mm[valid], tc, run["crop"], G)
            grid[meth][s:s + len(pb)] = gm.cpu().numpy().astype(np.float16)
        for b, p in enumerate(pb):
            boxes = [bx for bx in (box_cache_to_input(bx, cfg["data"]["cache_size"], tc, run["crop"]) for bx in p["boxes"]) if bx]
            mask = boxes_to_mask(boxes, S) if boxes else None
            for meth, mm in maps.items():
                m = mm[b]
                if torch.isnan(m[0, 0]):
                    continue
                r = dict(run_id=run["run_id"], pair_id=p["pair_id"], dataset=p["dataset"], target=p["target"],
                         method=meth, prob=float(probs[b]), box_visible=mask is not None)
                r.update(loc_metrics(m.cpu().numpy(), mask) if mask is not None else
                         dict(energy=np.nan, norm_energy=np.nan, pointing=np.nan, area_frac=0.0))
                rows.append(r)
                if p["in_faith"]:
                    am, ar, p0 = deletion_curve(model, x[b], p["idx"], m, fracs, rng)
                    frows.append(dict(run_id=run["run_id"], pair_id=p["pair_id"], dataset=p["dataset"],
                                      method=meth, del_auc_map=am, del_auc_random=ar, p0=p0,
                                      faith_gain=ar - am))
            if p["in_sanity"]:
                sanity_maps[s + b] = {meth: maps[meth][b].detach() for meth in maps}

    # ---- sanity check: randomizzazione ultimo stadio + testa ----------------
    srows = []
    if sanity_maps:
        ext.remove()
        rmodel = copy.deepcopy(model)
        randomize_modules(rmodel, last_block_and_head(rmodel), seed=0)
        rext = CamExtractor(rmodel, target_layer(rmodel), feature_layout(rmodel))
        keys = sorted(sanity_maps)
        for s in range(0, len(keys), a["cam_batch"]):
            ks = keys[s:s + a["cam_batch"]]
            pb = [pairs[k] for k in ks]
            x = torch.stack([tf(Image.open(p["cache_path"]).convert("RGB")) for p in pb]).to(device)
            rmaps, _ = compute_maps(rmodel, pb, x, cfg, rext, [True] * len(pb), base)
            for b, k in enumerate(ks):
                for meth in rmaps:
                    o = F.interpolate(sanity_maps[k][meth][None, None], size=(G, G), mode="area").flatten().cpu().numpy()
                    r = F.interpolate(rmaps[meth][b][None, None], size=(G, G), mode="area").flatten().cpu().numpy()
                    rho = spearmanr(o, r).statistic if np.std(o) > 0 and np.std(r) > 0 else np.nan
                    srows.append(dict(run_id=run["run_id"], pair_id=pairs[k]["pair_id"], method=meth,
                                      spearman_orig_vs_random=float(rho)))
        rext.remove()

    pd.DataFrame(rows).to_csv(os.path.join(rd, "attr_pairs.csv"), index=False)
    pd.DataFrame(frows).to_csv(os.path.join(rd, "attr_faithfulness.csv"), index=False)
    pd.DataFrame(srows).to_csv(os.path.join(rd, "attr_sanity.csv"), index=False)
    np.savez_compressed(os.path.join(rd, "attr_maps.npz"), pair_id=np.array([p["pair_id"] for p in pairs]),
                        **{f"map_{k}": v for k, v in grid.items()})
    summ = dict(run_id=run["run_id"], n_pairs=len(pairs))
    if rows:
        df = pd.DataFrame(rows)
        for (ds, meth), g in df.groupby(["dataset", "method"]):
            summ[f"{ds}_{meth}_pointing"] = float(g["pointing"].mean())
            summ[f"{ds}_{meth}_energy"] = float(g["energy"].mean())
    dump_json(summ, os.path.join(rd, "attr_done.json"))
    print(f"[DONE] attribuzione {run['run_id']}: {summ}")


if __name__ == "__main__":
    main()
