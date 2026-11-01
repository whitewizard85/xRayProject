"""
experiments.py -- piano sperimentale della versione A.

BLOCCO "factorial" (domanda principale, risponde al limite "7 condizioni su 16, 3 seed"):
    fattoriale COMPLETO 2^4 sui quattro fattori del paper
        crop     : 0 = resize 384 | 1 = resize 420 + center crop 384
        loss     : bce | asl
        schedule : fx10 = 10 epoche + cosine | es20 = fino a 20 epoche, early stopping (pazienza 8)
        aug      : 1 = flip orizzontale | 0 = nessuna augmentation
    16 celle x 5 seed = 80 run (ConvNeXt-Base IN-22k)

BLOCCO "losses" (generalizza il risultato di calibrazione oltre BCE vs ASL):
    cella di riferimento (crop=0, fx10, aug=1) con loss wbce (BCE con pos_weight) e focal (gamma=2)
    2 x 5 seed = 10 run   (bce e asl nella stessa cella sono gia' nel fattoriale)

BLOCCO "arch" (verifica che i risultati non siano specifici di ConvNeXt):
    DenseNet-121 nella cella di riferimento, loss bce e asl
    2 x 5 seed = 10 run

Totale: 100 run. L'ordine e' "seed piu' esterno": dopo i primi 20 run hai gia'
UNA replica completa di tutto, dopo 60 run tre repliche complete, ecc.
Cosi' i risultati parziali sono sempre bilanciati e analizzabili.
"""
import itertools

FACTOR_LEVELS = {"crop": [0, 1], "loss": ["bce", "asl"], "schedule": ["fx10", "es20"], "aug": [1, 0]}
REF_CELL = dict(crop=0, schedule="fx10", aug=1)


def make_run(block, arch, crop, loss, schedule, aug, seed):
    cond = f"{arch}_c{crop}_{loss}_{schedule}_a{aug}"
    return dict(run_id=f"{block}__{cond}__s{seed}", block=block, condition=cond, arch=arch,
                crop=int(crop), loss=loss, schedule=schedule, aug=int(aug), seed=int(seed))


def build_plan(cfg):
    seeds = cfg["experiment"]["seeds"]
    blocks = cfg["experiment"]["blocks"]
    plan = []
    for seed in seeds:
        if "factorial" in blocks:
            for crop, loss, sched, aug in itertools.product(*FACTOR_LEVELS.values()):
                plan.append(make_run("factorial", "convnext_base", crop, loss, sched, aug, seed))
        if "losses" in blocks:
            for loss in ["wbce", "focal"]:
                plan.append(make_run("losses", "convnext_base", REF_CELL["crop"], loss,
                                     REF_CELL["schedule"], REF_CELL["aug"], seed))
        if "arch" in blocks:
            for loss in ["bce", "asl"]:
                plan.append(make_run("arch", "densenet121", REF_CELL["crop"], loss,
                                     REF_CELL["schedule"], REF_CELL["aug"], seed))
    if cfg.get("_smoke"):
        # smoke: una cella per tipo, per esercitare ogni ramo del codice (crop, asl/bce/wbce/focal, es20, densenet)
        keep = {"factorial__convnext_base_c0_asl_fx10_a1", "factorial__convnext_base_c1_bce_es20_a0",
                "losses__convnext_base_c0_wbce_fx10_a1", "losses__convnext_base_c0_focal_fx10_a1",
                "arch__densenet121_c0_bce_fx10_a1"}
        plan = [r for r in plan if f"{r['block']}__{r['condition']}" in keep]
    return plan


def find_run(cfg, run_id):
    for r in build_plan(cfg):
        if r["run_id"] == run_id:
            return r
    raise KeyError(f"run_id sconosciuto: {run_id}")


if __name__ == "__main__":
    import argparse
    from common import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    plan = build_plan(load_config(a.config, a.smoke))
    for r in plan:
        print(r["run_id"])
    print(f"\nTotale run: {len(plan)}")
