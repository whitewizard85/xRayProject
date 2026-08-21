"""
analyze_bbox_results.py

OBIETTIVO
---------
Applica alla metrica bbox pointing-game / energy-in-bbox lo stesso
trattamento statistico rigoroso gia' usato per la metrica body-silhouette
(bootstrap patient-clustered E gerarchico seed+paziente), per verificare se
il vantaggio di C6 sul pointing-game (visibile nei numeri grezzi, zero
sovrapposizione fra i 3 seed di C0 e i 3 seed di C6) sopravvive a un
trattamento statistico onesto, o se e' anch'esso un artefatto di piccolo
campione di seed.

USO
---
    python analyze_bbox_results.py
    python analyze_bbox_results.py --conditions C0_baseline,C6_full_combined

Richiede i file bbox_localization_<condizione>_seed<seed>.csv gia' prodotti
da bbox_pointing_game_localization.py, e Data_Entry_2017.csv per la mappa
paziente.
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUTPUT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/ablation_results"
NIH_METADATA_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/archive/Data_Entry_2017.csv"
N_BOOTSTRAP = 10000
CI_ALPHA = 0.05
BASELINE_CONDITION = "C0_baseline"


def load_patient_map():
    meta = pd.read_csv(NIH_METADATA_CSV)
    return dict(zip(meta["Image Index"], meta["Patient ID"]))


def hierarchical_bootstrap(diffs, patients, seeds, n_boot=N_BOOTSTRAP, seed_rng=42):
    """Identico a quello usato per la metrica body-silhouette: ricampiona
    prima i seed, poi i pazienti dentro ciascun seed ricampionato."""
    rng = np.random.default_rng(seed_rng)
    df = pd.DataFrame({"diff": diffs, "patient": patients, "seed": seeds}).dropna()
    seeds_available = df["seed"].unique()
    seed_patient_groups = {
        s: {p: g["diff"].to_numpy() for p, g in df[df["seed"] == s].groupby("patient")}
        for s in seeds_available
    }
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sampled_seeds = rng.choice(seeds_available, size=len(seeds_available), replace=True)
        pieces = []
        for s in sampled_seeds:
            pd_ = seed_patient_groups[s]
            pats = np.array(list(pd_.keys()))
            sampled_pats = rng.choice(pats, size=len(pats), replace=True)
            pieces.extend(pd_[p] for p in sampled_pats)
        boot_means[i] = np.concatenate(pieces).mean()
    lo, hi = np.percentile(boot_means, [100 * CI_ALPHA / 2, 100 * (1 - CI_ALPHA / 2)])
    return df["diff"].mean(), lo, hi


def patient_clustered_bootstrap(values, patient_ids, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"value": values, "patient": patient_ids}).dropna()
    grouped = {p: g["value"].to_numpy() for p, g in df.groupby("patient")}
    unique_patients = np.array(list(grouped.keys()))
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        boot_means[i] = np.concatenate([grouped[p] for p in sampled]).mean()
    lo, hi = np.percentile(boot_means, [100 * CI_ALPHA / 2, 100 * (1 - CI_ALPHA / 2)])
    return df["value"].mean(), lo, hi


def load_all(conditions, seeds):
    frames = []
    for cond in conditions:
        for seed in seeds:
            path = os.path.join(OUTPUT_DIR, f"bbox_localization_{cond}_seed{seed}.csv")
            if os.path.exists(path):
                frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("Nessun file bbox_localization_*.csv trovato -- lancia prima "
                                 "bbox_pointing_game_localization.py")
    return pd.concat(frames, ignore_index=True)


def analyze_metric(all_df, metric_col, conditions, patient_map):
    baseline = all_df[all_df["condition"] == BASELINE_CONDITION]
    results = []
    for cond in conditions:
        if cond == BASELINE_CONDITION:
            continue
        cond_df = all_df[all_df["condition"] == cond]
        # IMPORTANTE: il join include "class" oltre a Image Index e seed.
        # Senza "class", un'immagine con piu' di una patologia annotata
        # (righe multiple con la stessa Image Index) produce un cross-join
        # spurio fra tutte le combinazioni di classi invece di accoppiare
        # ogni annotazione con la sua controparte della stessa classe,
        # gonfiando il numero di coppie e mescolando confronti fra classi
        # diverse sulla stessa immagine.
        merged = pd.merge(
            baseline[["Image Index", "seed", "class", metric_col]],
            cond_df[["Image Index", "seed", "class", metric_col]],
            on=["Image Index", "seed", "class"], suffixes=("_base", "_cond")
        ).dropna()
        if len(merged) == 0:
            continue
        diffs = merged[f"{metric_col}_cond"] - merged[f"{metric_col}_base"]
        merged["patient"] = merged["Image Index"].map(patient_map)

        try:
            stat, pval = wilcoxon(merged[f"{metric_col}_cond"], merged[f"{metric_col}_base"])
        except ValueError:
            pval = np.nan

        mean_p, ci_p_lo, ci_p_hi = patient_clustered_bootstrap(diffs.values, merged["patient"].values)
        mean_h, ci_h_lo, ci_h_hi = hierarchical_bootstrap(diffs.values, merged["patient"].values, merged["seed"].values)

        results.append({
            "metric": metric_col, "condition": cond, "n_pairs": len(merged),
            "n_unique_patients": merged["patient"].nunique(),
            "mean_diff": mean_p, "median_diff": diffs.median(),
            "pct_improved": (diffs > 0).mean() * 100,
            "CI_95_lo_PATIENT_ONLY": ci_p_lo, "CI_95_hi_PATIENT_ONLY": ci_p_hi,
            "CI_95_lo_HIERARCHICAL": ci_h_lo, "CI_95_hi_HIERARCHICAL": ci_h_hi,
            "wilcoxon_p": pval,
        })
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=str, default="C0_baseline,C6_full_combined")
    parser.add_argument("--seeds", type=str, default="42,123,2026")
    args = parser.parse_args()
    conditions = args.conditions.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    all_df = load_all(conditions, seeds)
    patient_map = load_patient_map()

    pg_res = analyze_metric(all_df, "pointing_game_hit", conditions, patient_map)
    en_res = analyze_metric(all_df, "energy_in_bbox", conditions, patient_map)
    combined = pd.concat([pg_res, en_res], ignore_index=True)

    out_path = os.path.join(OUTPUT_DIR, "SUMMARY_bbox_statistical_analysis.csv")
    combined.to_csv(out_path, index=False)
    print(f"\n{'='*100}\nBBOX METRICS -- confronto statistico C0 vs condizioni (patient-only e gerarchico)\n{'='*100}")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(combined.to_string(index=False))
    print(f"\nFull table: {out_path}")
    print("\n[COME LEGGERE]")
    print("- Se il CI HIERARCHICAL esclude lo zero per pointing_game_hit, il vantaggio del")
    print("  pointing-game e' statisticamente difendibile anche tenendo conto della sola")
    print("  variabilita' fra 3 seed -- un risultato molto piu' solido del confronto")
    print("  sulla metrica body-silhouette, che non sopravvive allo stesso trattamento.")


if __name__ == "__main__":
    main()
