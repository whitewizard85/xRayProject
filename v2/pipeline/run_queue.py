"""
run_queue.py -- esegue TUTTO il piano in continuo, in modo ripristinabile.

    python run_queue.py                 # piano completo (100 run)
    python run_queue.py --smoke         # collaudo rapido end-to-end (5 run mini)
    python run_queue.py --stages train  # solo training (es. per anticipare la GPU)
    python run_queue.py --stages attrx  # solo attribuzione estesa (cascata + IG-blur) sui run gia' fatti

Per ogni run esegue in sequenza 03_train -> 04_evaluate -> 05_attribution, ciascuno
in un SOTTOPROCESSO (contesto CUDA pulito a ogni stadio, niente memory leak).
- Gli stadi gia' completati vengono saltati (marker *_done.json / eval.json).
- Se uno stadio fallisce, l'errore va in runs/<id>/error_<stage>.log e la coda passa
  al run successivo; i falliti vengono ritentati a fine giro (max --max-attempts volte).
- Un lock impedisce di avere due code attive contemporaneamente.
- Ogni --analyze-every run completati lancia 06_analyze.py, cosi' i risultati
  parziali sono sempre aggiornati in <work_dir>/analysis/.
"""
import os
import sys
import json
import time
import fcntl
import argparse
import subprocess
import datetime

from common import load_config, run_dir, dump_json
from experiments import build_plan

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = [("train", "03_train.py", "train_done.json"),
          ("eval", "04_evaluate.py", "eval.json"),
          ("attr", "05_attribution.py", "attr_done.json"),
          ("attrx", "05b_attribution_ext.py", "attr_ext_done.json")]   # estensione 2026-11


def log(msg, fh):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def run_stage(cfg, args, run, stage, script, logf):
    rd = run_dir(cfg, run["run_id"])
    cmd = [sys.executable, os.path.join(HERE, script), "--run", run["run_id"]]
    if args.config:
        cmd += ["--config", args.config]
    if args.smoke:
        cmd += ["--smoke"]
    out_path = os.path.join(rd, f"stdout_{stage}.log")
    t0 = time.time()
    with open(out_path, "a") as out:
        out.write(f"\n===== {datetime.datetime.now()} {' '.join(cmd)}\n")
        out.flush()
        rc = subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=HERE)
    dt = (time.time() - t0) / 3600
    if rc != 0:
        with open(out_path) as f:
            tail = f.readlines()[-60:]
        with open(os.path.join(rd, f"error_{stage}.log"), "a") as ef:
            ef.write(f"\n===== {datetime.datetime.now()} exit code {rc}\n" + "".join(tail))
        log(f"  FALLITO {stage} {run['run_id']} (rc={rc}, {dt:.2f} h) -> vedi {rd}/error_{stage}.log", logf)
        return False
    log(f"  ok {stage} {run['run_id']} ({dt:.2f} h)", logf)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stages", default="train,eval,attr,attrx")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--analyze-every", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(args.config, args.smoke)
    wd = cfg["paths"]["work_dir"]
    lock = open(os.path.join(wd, "queue.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Un'altra coda e' gia' in esecuzione su questa work_dir. Esco.")
        sys.exit(0)
    logf = open(os.path.join(wd, "queue.log"), "a")
    wanted = args.stages.split(",")
    plan = build_plan(cfg)
    attempts_p = os.path.join(wd, "queue_attempts.json")
    attempts = json.load(open(attempts_p)) if os.path.exists(attempts_p) else {}
    log(f"Coda avviata: {len(plan)} run, stadi={wanted}, smoke={args.smoke}", logf)

    completed_since_analysis = 0
    while True:
        progressed, pending = False, 0
        for i, run in enumerate(plan):
            rd = run_dir(cfg, run["run_id"])
            for si, (stage, script, marker) in enumerate(STAGES):
                if stage not in wanted or os.path.exists(os.path.join(rd, marker)):
                    continue
                # prerequisiti non soddisfatti e non richiesti in questa coda (es. --stages attrx su run non
                # ancora addestrati): salta senza consumare tentativi
                if any(not os.path.exists(os.path.join(rd, m)) for st, _, m in STAGES[:si] if st not in wanted):
                    break
                key = f"{run['run_id']}:{stage}"
                if attempts.get(key, 0) >= args.max_attempts:
                    break   # stadio rinunciato: non eseguire i successivi di questo run
                pending += 1
                log(f"[{i + 1}/{len(plan)}] {stage} {run['run_id']}", logf)
                ok = run_stage(cfg, args, run, stage, script, logf)
                if not ok:
                    attempts[key] = attempts.get(key, 0) + 1
                    dump_json(attempts, attempts_p)
                    break
                progressed = True
                if stage == wanted[-1]:
                    completed_since_analysis += 1
            if completed_since_analysis >= args.analyze_every:
                subprocess.call([sys.executable, os.path.join(HERE, "06_analyze.py")] +
                                (["--config", args.config] if args.config else []) + (["--smoke"] if args.smoke else []),
                                cwd=HERE)
                completed_since_analysis = 0
        if pending == 0:   # ogni stadio pendente o riesce o consuma un tentativo -> termina sempre
            break
        if not progressed:
            time.sleep(60)  # solo fallimenti in questo giro: breve pausa prima di ritentare
    subprocess.call([sys.executable, os.path.join(HERE, "06_analyze.py")] +
                    (["--config", args.config] if args.config else []) + (["--smoke"] if args.smoke else []), cwd=HERE)
    given_up = [k for k, v in attempts.items() if v >= args.max_attempts]
    log(f"Coda terminata. Stadi abbandonati dopo {args.max_attempts} tentativi: {given_up or 'nessuno'}", logf)


if __name__ == "__main__":
    main()
