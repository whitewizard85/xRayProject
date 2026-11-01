"""
status.py -- stato di avanzamento della coda e stima del tempo residuo.

    python status.py
"""
import os
import json
import argparse
import datetime

from common import load_config
from experiments import build_plan

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
cfg = load_config(a.config, a.smoke)
plan = build_plan(cfg)
wd = cfg["paths"]["work_dir"]
cnt = {"train": 0, "eval": 0, "attr": 0}
hours, failed, running = [], [], []
for r in plan:
    rd = os.path.join(wd, "runs", r["run_id"])
    if os.path.exists(os.path.join(rd, "train_done.json")):
        cnt["train"] += 1
        hours.append(json.load(open(os.path.join(rd, "train_done.json")))["total_hours"])
    elif os.path.exists(os.path.join(rd, "last.pt")):
        running.append(r["run_id"])
    if os.path.exists(os.path.join(rd, "eval.json")):
        cnt["eval"] += 1
    if os.path.exists(os.path.join(rd, "attr_done.json")):
        cnt["attr"] += 1
    if any(f.startswith("error_") for f in (os.listdir(rd) if os.path.isdir(rd) else [])):
        failed.append(r["run_id"])
n = len(plan)
print(f"Piano: {n} run | train {cnt['train']}/{n} | eval {cnt['eval']}/{n} | attr {cnt['attr']}/{n}")
if running:
    print("In corso / interrotti (ripartiranno):", ", ".join(running))
if hours:
    avg = sum(hours) / len(hours)
    rem = (n - cnt["train"]) * (avg + 0.4)     # +0.4 h circa per eval+attribuzione (stima grezza)
    print(f"Ore di training medie per run: {avg:.2f} | stima residua: {rem:.0f} h "
          f"(~{rem / 24:.1f} giorni, fine ~{datetime.datetime.now() + datetime.timedelta(hours=rem):%d/%m %H:%M})")
if failed:
    print("Run con almeno un errore registrato (vedi runs/<id>/error_*.log):")
    for f in failed:
        print("  -", f)
