#!/usr/bin/env python3
"""Raccoglie i numeri mancanti per il paper. Sola lettura: non modifica nulla.

Uso (sul server):  python 07_paper_numbers.py [config.yaml]
Output:            <work_dir>/paper_numbers/  (+ paper_numbers.tgz da inviare)
  - best_epoch_per_run.csv    epoca selezionata (checkpoint migliore) per ogni run
  - best_epoch_summary.txt    riepilogo per schedule
  - perclass_auc_runs.csv     AUC per classe e per run (NIH test e VinDr)
  - perclass_auc_summary.csv  media, SD tra seed e IC95% (bootstrap sui run) per classe, NIH vs VinDr
  - timing.txt                ore di training totali (da train_done.json)
Contiene solo statistiche aggregate: nessun dato VinDr per immagine.
"""
import glob
import json
import os
import sys
import tarfile

import numpy as np
import pandas as pd
import yaml

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config.yaml"))
work = cfg["paths"]["work_dir"]
runs_root = os.path.join(work, "runs")
out = os.path.join(work, "paper_numbers")
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- epoche selezionate
rows = []
for f in sorted(glob.glob(os.path.join(runs_root, "*", "train_done.json"))):
    d = json.load(open(f))
    r, h = d["run"], d["history"]
    aucs = [e["val_macro_auc"] for e in h]
    best_ep = h[int(np.argmax(aucs))]["epoch"]
    rows.append(dict(run_id=r["run_id"], arch=r.get("arch"), crop=r.get("crop"), loss=r.get("loss"),
                     schedule=r.get("schedule"), aug=r.get("aug"), seed=r.get("seed"),
                     epochs_run=len(h), best_epoch=best_ep, best_val_auc=max(aucs),
                     val_auc_last=aucs[-1], hours=d.get("total_hours", np.nan)))
be = pd.DataFrame(rows)
be.to_csv(os.path.join(out, "best_epoch_per_run.csv"), index=False)

with open(os.path.join(out, "best_epoch_summary.txt"), "w") as fh:
    fh.write(f"run trovati: {len(be)}\n\n")
    for s, g in be.groupby("schedule"):
        fh.write(f"== schedule {s} (n={len(g)}) ==\n")
        fh.write("distribuzione best_epoch:\n" + g["best_epoch"].value_counts().sort_index().to_string() + "\n")
        fh.write(f"mediana {g.best_epoch.median()}, min {g.best_epoch.min()}, max {g.best_epoch.max()}\n")
        fh.write(f"calo medio val AUC (best - ultima epoca): {(g.best_val_auc - g.val_auc_last).mean():.4f}\n\n")
    # coppie same-seed fx10/es20: stessa epoca selezionata?
    key = ["arch", "crop", "loss", "aug", "seed"]
    p = be[be.schedule == "fx10"].merge(be[be.schedule == "es20"], on=key, suffixes=("_fx", "_es"))
    if len(p):
        fh.write(f"coppie fx10/es20 con lo stesso seed: {len(p)}\n")
        fh.write(f"  stessa best_epoch: {(p.best_epoch_fx == p.best_epoch_es).sum()}\n")
        fh.write(f"  |diff best_epoch| mediana: {(p.best_epoch_fx - p.best_epoch_es).abs().median()}\n")

# ---------------------------------------------------------------- AUC per classe
pcs = []
for f in sorted(glob.glob(os.path.join(runs_root, "*", "eval_perclass.csv"))):
    pcs.append(pd.read_csv(f))
pc = pd.concat(pcs, ignore_index=True)
cols = [c for c in ["run_id", "block", "arch", "crop", "loss", "schedule", "aug", "seed",
                    "dataset", "target", "n", "n_pos", "auc", "pr_auc"] if c in pc.columns]
pc[cols].to_csv(os.path.join(out, "perclass_auc_runs.csv"), index=False)


def boot_ci(x, B=5000):
    x = np.asarray(x, float)
    m = rng.choice(x, (B, len(x)), replace=True).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


summ = []
groups = {
    "convnext_factorial": (pc["arch"] == "convnext_base") & (pc["block"] == "factorial") if "block" in pc else None,
    "convnext_all": pc["arch"] == "convnext_base",
    "densenet121": pc["arch"] == "densenet121",
}
for gname, mask in groups.items():
    if mask is None:
        continue
    sub = pc[mask]
    for (ds, tgt), g in sub.groupby(["dataset", "target"]):
        # media per seed prima (schedule collassato): unita' indipendenti
        if "schedule" in g and g["schedule"].nunique() > 1:
            k = [c for c in ["crop", "loss", "aug", "seed"] if c in g]
            vals = g.groupby(k)["auc"].mean().values
        else:
            vals = g["auc"].values
        lo, hi = boot_ci(vals)
        summ.append(dict(group=gname, dataset=ds, target=tgt, n_pos=int(g["n_pos"].iloc[0]),
                         units=len(vals), auc_mean=vals.mean(), auc_sd=vals.std(ddof=1),
                         ci95_lo=lo, ci95_hi=hi))
sm = pd.DataFrame(summ)
sm.to_csv(os.path.join(out, "perclass_auc_summary.csv"), index=False, float_format="%.4f")

# confronto NIH vs VinDr sulle classi comuni
nih = sm[sm.dataset == "nih_test"][["group", "target", "auc_mean"]].rename(columns={"auc_mean": "nih"})
vd = sm[sm.dataset == "vindr_test"][["group", "target", "auc_mean", "ci95_lo", "ci95_hi", "n_pos"]]
cmp_ = vd.merge(nih, on=["group", "target"])
cmp_["delta_vindr_minus_nih"] = cmp_["auc_mean"] - cmp_["nih"]
cmp_.sort_values(["group", "delta_vindr_minus_nih"], ascending=[True, False]).to_csv(
    os.path.join(out, "perclass_nih_vs_vindr.csv"), index=False, float_format="%.4f")

# ---------------------------------------------------------------- tempi
with open(os.path.join(out, "timing.txt"), "w") as fh:
    fh.write(f"ore di training totali: {be.hours.sum():.1f}\n")
    fh.write(be.groupby("arch").hours.agg(["count", "sum", "mean"]).to_string() + "\n")

with tarfile.open(os.path.join(work, "paper_numbers.tgz"), "w:gz") as t:
    t.add(out, arcname="paper_numbers")
print(open(os.path.join(out, "best_epoch_summary.txt")).read())
print(cmp_[cmp_.group == "convnext_factorial"].to_string(index=False))
print(f"\nCreato {os.path.join(work, 'paper_numbers.tgz')}")
