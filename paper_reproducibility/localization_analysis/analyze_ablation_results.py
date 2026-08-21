"""
analyze_ablation_results.py

OBIETTIVO
---------
Prende gli output di ablation_train_and_eval.py (metriche globali per
condizione/seed + metrica di localizzazione per condizione/seed) e produce
l'analisi statistica che il paper attualmente non ha:

  1. Media +/- deviazione standard fra seed per ogni condizione, su tutte
     le metriche globali (Macro/Micro AUC, PR-AUC, Precision, Recall, F1).
  2. Per la metrica di localizzazione: mediana, IQR, effect size
     (rank-biserial), intervallo di confidenza bootstrap CLUSTERIZZATO PER
     PAZIENTE (non per immagine), per isolare l'effetto di ciascun fattore
     (crop, loss, schedule) confrontato con la baseline C0.
  3. Un confronto esplicito: la baseline C0 ha piu' pazienti unici del
     campione di 1000 immagini? (Se un paziente compare piu' volte nel
     campione, le sue immagini non sono indipendenti -- il bootstrap
     clusterizzato lo tiene in conto, ricampionando per paziente intero
     invece che per immagine.)

Lancialo DOPO che ablation_train_and_eval.py ha completato almeno la
baseline (C0) e la condizione che vuoi confrontare. Puoi rilanciarlo man
mano che altre condizioni finiscono, senza aspettare tutto il fattoriale.

USO
---
    python analyze_ablation_results.py

Richiede il file Data_Entry_2017.csv originale del dataset NIH (per
mappare Image Index -> Patient ID, necessario per il clustering).
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ============================================================
# CONFIGURAZIONE
# ============================================================
OUTPUT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/ablation_results"
NIH_METADATA_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/archive/Data_Entry_2017.csv"
N_BOOTSTRAP = 10000
BASELINE_CONDITION = "C0_baseline"
CI_ALPHA = 0.05  # -> intervallo di confidenza al 95%
USE_FIXED_MASK = True  # True = analizza i file *_FIXEDMASK.csv (maschera corretta, spalla inclusa)
                        # False = analizza i vecchi file (maschera con il bug della spalla esclusa)

CONDITIONS_ORDER = [
    "C0_baseline", "C1_crop_only", "C2_bce_only", "C3_schedule_only",
    "C4_noaug_only", "C5_crop_bce", "C6_full_combined",
]


def load_patient_map():
    """Image Index -> Patient ID, per il bootstrap clusterizzato."""
    meta = pd.read_csv(NIH_METADATA_CSV)
    return dict(zip(meta["Image Index"], meta["Patient ID"]))


def summarize_global_metrics():
    path = os.path.join(OUTPUT_DIR, "global_metrics.csv")
    if not os.path.exists(path):
        print(f"[WARN] {path} not found -- run training first.")
        return None
    df = pd.read_csv(path)
    metrics_cols = ["macro_auc", "micro_auc", "macro_prauc", "precision", "recall", "f1"]

    rows = []
    for cond in CONDITIONS_ORDER:
        sub = df[df["condition"] == cond]
        if len(sub) == 0:
            continue
        row = {"condition": cond, "n_seeds": len(sub)}
        for m in metrics_cols:
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_std"] = sub[m].std(ddof=1) if len(sub) > 1 else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "SUMMARY_global_metrics.csv")
    summary.to_csv(out_path, index=False)
    print(f"\n{'='*90}\nGLOBAL METRICS -- mean +/- std across seeds, per condition\n{'='*90}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[["condition", "n_seeds", "macro_auc_mean", "macro_auc_std",
                        "macro_prauc_mean", "macro_prauc_std"]].to_string(index=False))
    print(f"\nFull table written to {out_path}")
    return summary


def hierarchical_bootstrap(diffs, patients, seeds, n_boot=N_BOOTSTRAP, seed_rng=42):
    """
    Bootstrap a DUE LIVELLI (seed poi paziente), invece del solo bootstrap
    per paziente usato in patient_clustered_bootstrap(). Le 3000 osservazioni
    (1000 immagini x 3 seed) NON sono 3000 repliche indipendenti: derivano da
    soli 3 modelli allenati indipendentemente (i seed). Il bootstrap per solo
    paziente ignora questo e rischia di sottostimare l'incertezza vera.

    Qui si ricampiona PRIMA il livello seed (con reinserimento, dai seed
    disponibili), POI, per ciascun seed ricampionato, si ricampionano i
    pazienti con reinserimento SOLO dentro i dati di quel seed. Questo
    propaga correttamente sia la variabilita' fra modelli (seed) sia quella
    fra pazienti dentro ciascun modello.

    LIMITE ONESTO: con solo 3 seed disponibili, il livello di ricampionamento
    "seed" e' statisticamente grezzo (solo 3^3=27 combinazioni possibili). Il
    risultato e' comunque piu' onesto del bootstrap a un solo livello, ma se
    hai tempo per allenare 2 seed aggiuntivi (5 totali), l'intervallo
    risultante sarebbe piu' affidabile -- lo segnaliamo esplicitamente nel
    paper piuttosto che nasconderlo.
    """
    rng = np.random.default_rng(seed_rng)
    df = pd.DataFrame({"diff": diffs, "patient": patients, "seed": seeds}).dropna()
    seeds_available = df["seed"].unique()

    # pre-raggruppa una sola volta: seed -> {paziente -> array di diff}
    seed_patient_groups = {}
    for s in seeds_available:
        sub = df[df["seed"] == s]
        seed_patient_groups[s] = {p: g["diff"].to_numpy() for p, g in sub.groupby("patient")}

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sampled_seeds = rng.choice(seeds_available, size=len(seeds_available), replace=True)
        pieces = []
        for s in sampled_seeds:
            patient_dict = seed_patient_groups[s]
            patients_in_seed = np.array(list(patient_dict.keys()))
            sampled_patients = rng.choice(patients_in_seed, size=len(patients_in_seed), replace=True)
            pieces.extend(patient_dict[p] for p in sampled_patients)
        boot_means[i] = np.concatenate(pieces).mean()

    lo, hi = np.percentile(boot_means, [100 * CI_ALPHA / 2, 100 * (1 - CI_ALPHA / 2)])
    return df["diff"].mean(), lo, hi, len(seeds_available)


def patient_clustered_bootstrap(values, patient_ids, n_boot=N_BOOTSTRAP, seed=42):
    """
    Ricampiona per paziente (con reinserimento), non per immagine: se un
    paziente ha piu' immagini, tutte le sue immagini vengono incluse o
    escluse insieme in ogni replica bootstrap. Questo e' il modo corretto
    di stimare un intervallo di confidenza quando le osservazioni non sono
    indipendenti a livello immagine (piu' immagini per paziente).

    Implementazione efficiente: raggruppa i valori per paziente UNA sola
    volta (dict paziente -> array), poi ogni replica bootstrap fa solo
    lookup + concatenazione (niente filtri ripetuti su DataFrame).
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"value": values, "patient": patient_ids}).dropna()
    grouped = {p: g["value"].to_numpy() for p, g in df.groupby("patient")}
    unique_patients = np.array(list(grouped.keys()))

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        sampled_vals = np.concatenate([grouped[p] for p in sampled_patients])
        boot_means[i] = sampled_vals.mean()

    lo, hi = np.percentile(boot_means, [100 * CI_ALPHA / 2, 100 * (1 - CI_ALPHA / 2)])
    return df["value"].mean(), lo, hi, len(unique_patients)


def rank_biserial_effect_size(x, y):
    """Effect size non parametrico appaiato, coerente col test di Wilcoxon."""
    diff = np.array(x) - np.array(y)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return np.nan
    n_pos = np.sum(diff > 0)
    n_neg = np.sum(diff < 0)
    return (n_pos - n_neg) / len(diff)


def analyze_localization():
    patient_map = load_patient_map() if os.path.exists(NIH_METADATA_CSV) else None
    if patient_map is None:
        print(f"[WARN] {NIH_METADATA_CSV} not found -- patient-clustered CI will be skipped "
              f"(falling back to image-level bootstrap, which UNDERSTATES the true uncertainty).")

    if USE_FIXED_MASK:
        loc_files = [f for f in os.listdir(OUTPUT_DIR)
                     if f.startswith("localization_") and f.endswith("_FIXEDMASK.csv")]
        print(f"[INFO] USE_FIXED_MASK=True -> uso solo i file *_FIXEDMASK.csv ({len(loc_files)} trovati)")
    else:
        loc_files = [f for f in os.listdir(OUTPUT_DIR)
                     if f.startswith("localization_") and f.endswith(".csv") and "FIXEDMASK" not in f]
        print(f"[INFO] USE_FIXED_MASK=False -> uso i file con la maschera originale ({len(loc_files)} trovati)")
    if not loc_files:
        print("[WARN] No localization_*.csv files found -- run training/localization first.")
        return

    all_loc = pd.concat([pd.read_csv(os.path.join(OUTPUT_DIR, f)) for f in loc_files], ignore_index=True)

    # media per condizione, aggregata sui seed disponibili (una riga per immagine x condizione x seed)
    print(f"\n{'='*90}\nLOCALIZATION METRIC -- baseline ({BASELINE_CONDITION}) vs each condition\n{'='*90}")

    baseline = all_loc[all_loc["condition"] == BASELINE_CONDITION]
    results = []
    for cond in CONDITIONS_ORDER:
        if cond == BASELINE_CONDITION:
            continue
        cond_df = all_loc[all_loc["condition"] == cond]
        if len(cond_df) == 0:
            continue

        # media per seed, poi confronto seed-aggregato (piu' seed = piu' potenza,
        # ma qui usiamo il pooling per-immagine su tutti i seed disponibili come
        # prima approssimazione -- per un'analisi mixed-effects seed+paziente
        # rigorosa, esporta questi CSV e usa un modello a effetti misti)
        merged = pd.merge(
            baseline[["Image Index", "fraction_inside", "seed"]],
            cond_df[["Image Index", "fraction_inside", "seed"]],
            on=["Image Index", "seed"], suffixes=("_baseline", "_cond")
        )
        if len(merged) == 0:
            continue

        diffs = merged["fraction_inside_cond"] - merged["fraction_inside_baseline"]
        median_diff = diffs.median()
        iqr = diffs.quantile(0.75) - diffs.quantile(0.25)
        pct_improved = (diffs > 0).mean() * 100
        effect_size = rank_biserial_effect_size(merged["fraction_inside_cond"], merged["fraction_inside_baseline"])

        try:
            stat, pval = wilcoxon(merged["fraction_inside_cond"], merged["fraction_inside_baseline"])
        except ValueError:
            stat, pval = np.nan, np.nan

        if patient_map is not None:
            merged["patient"] = merged["Image Index"].map(patient_map)
            mean_diff, ci_lo, ci_hi, n_patients = patient_clustered_bootstrap(diffs.values, merged["patient"].values)
            _, hier_ci_lo, hier_ci_hi, n_seeds_used = hierarchical_bootstrap(
                diffs.values, merged["patient"].values, merged["seed"].values)
        else:
            mean_diff, ci_lo, ci_hi = diffs.mean(), np.nan, np.nan
            hier_ci_lo, hier_ci_hi, n_seeds_used = np.nan, np.nan, np.nan
            n_patients = merged["Image Index"].nunique()

        results.append({
            "condition": cond, "n_pairs": len(merged), "n_unique_patients": n_patients,
            "mean_diff_vs_baseline": mean_diff, "median_diff": median_diff, "IQR": iqr,
            "pct_images_improved": pct_improved, "rank_biserial_effect_size": effect_size,
            "bootstrap_CI_95_lo_PATIENT_ONLY": ci_lo, "bootstrap_CI_95_hi_PATIENT_ONLY": ci_hi,
            "bootstrap_CI_95_lo_HIERARCHICAL": hier_ci_lo, "bootstrap_CI_95_hi_HIERARCHICAL": hier_ci_hi,
            "n_seeds_used": n_seeds_used,
            "wilcoxon_p": pval,
        })

    res_df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "SUMMARY_localization_vs_baseline.csv")
    res_df.to_csv(out_path, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(res_df.to_string(index=False))
    print(f"\nFull table written to {out_path}")

    print("\n[HOW TO READ THIS TABLE]")
    print("- 'mean_diff_vs_baseline' with a PATIENT_ONLY 95% CI that excludes zero indicates a change")
    print("  distinguishable from noise for THAT SPECIFIC factor combination, but this interval treats")
    print("  all seeds as extra independent images per patient rather than as separate model instances.")
    print("- The HIERARCHICAL CI (resamples seeds, then patients within each resampled seed) is the more")
    print("  honest interval: it is wider than PATIENT_ONLY whenever seed-to-seed variability matters,")
    print("  and is the one to report as the primary interval in the paper. With only 3 seeds this level")
    print("  of the bootstrap is statistically coarse (27 possible seed combinations) -- if you have time")
    print("  to train 2 more seeds (5 total), rerun this for a more reliable hierarchical interval.")
    print("- Compare C1_crop_only, C2_bce_only, C3_schedule_only individually against C0:")
    print("  whichever shows the largest, most confident shift is the dominant factor.")
    print("- C6_full_combined vs C0 reproduces the original paper's headline comparison,")
    print("  now with proper uncertainty quantification instead of a bare p-value.")
    return res_df


def main():
    summarize_global_metrics()
    analyze_localization()
    print("\n[NEXT STEP] Paste the SUMMARY_*.csv tables into the paper's Results/Discussion,")
    print("replacing the single-run, uncontrolled comparison currently reported there.")


if __name__ == "__main__":
    main()
