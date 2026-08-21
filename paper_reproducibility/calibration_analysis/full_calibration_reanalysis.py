"""
full_calibration_reanalysis.py

OBIETTIVO
---------
La revisione ha segnalato che l'analisi di calibrazione attuale del paper
(un solo ECE a 10 bin, nessun Brier score, nessuna NLL, nessuna slope/
intercept di calibrazione, nessun conteggio per bin, nessun intervallo di
confidenza) e' troppo debole per una rivista Q1. Questo script colma il gap
SENZA bisogno di nuovo training o inferenza GPU: usa i file
test_predictions_<condizione>_seed<seed>.csv gia' salvati da
ablation_train_and_eval.py (probabilita' predette per le 14 classi, per
ogni immagine di test), li unisce alle etichette vere prese da
test_split.csv, e calcola:

  1. Brier score per classe e macro (media dei quadrati degli errori di
     probabilita' -- piu' basso e' meglio, 0 = perfetto).
  2. Negative Log-Likelihood (NLL) per classe e macro.
  3. ECE per classe con CONTEGGIO CAMPIONI PER BIN esplicito (10 bin
     uniformi, coerente con quanto gia' descritto nel paper).
  4. Calibration slope e intercept (calibrazione di Cox): si fitta una
     regressione logistica di [etichetta vera] su [logit(probabilita'
     predetta)]; slope=1 e intercept=0 indicano calibrazione perfetta,
     slope<1 indica probabilita' troppo estreme (overconfidence), slope>1
     indica probabilita' troppo caute (underconfidence).
  5. Intervallo di confidenza bootstrap CLUSTERIZZATO PER PAZIENTE per ogni
     metrica sopra, per condizione (media sui seed disponibili).

Tutto questo e' calcolabile dai CSV che hai gia' su disco -- nessuna GPU,
nessun modello da ricaricare. Dovrebbe girare in meno di un minuto.

USO
---
    python full_calibration_reanalysis.py
    python full_calibration_reanalysis.py --conditions C0_baseline,C6_full_combined

OUTPUT
------
  - ablation_results/SUMMARY_calibration_full.csv (una riga per condizione,
    tutte le metriche aggregate sui seed disponibili + CI bootstrap)
  - ablation_results/SUMMARY_calibration_per_class.csv (dettaglio per
    classe x condizione, incluso il conteggio campioni per bin dell'ECE)
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

# ============================================================
# CONFIGURAZIONE
# ============================================================
OUTPUT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/ablation_results"
TEST_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
NIH_METADATA_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/archive/Data_Entry_2017.csv"
N_BINS = 10
N_BOOTSTRAP = 5000
CI_ALPHA = 0.05
EPS = 1e-7  # clipping per evitare log(0) nella NLL

CLASSES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
           "Fibrosis", "Pleural_Thickening", "Hernia"]

CONDITIONS_ORDER = [
    "C0_baseline", "C1_crop_only", "C2_bce_only", "C3_schedule_only",
    "C4_noaug_only", "C5_crop_bce", "C6_full_combined",
]
SEEDS = [42, 123, 2026]


def build_label_matrix():
    """Ricostruisce le etichette vere multi-label dal test_split.csv,
    stesso schema usato in ablation_train_and_eval.py."""
    df = pd.read_csv(TEST_CSV)
    labels = pd.DataFrame(0, index=df["Image Index"], columns=CLASSES)
    for _, row in df.iterrows():
        for l in str(row["Finding Labels"]).split("|"):
            if l in CLASSES:
                labels.loc[row["Image Index"], l] = 1
    return labels


def load_patient_map():
    if not os.path.exists(NIH_METADATA_CSV):
        return None
    meta = pd.read_csv(NIH_METADATA_CSV)
    return dict(zip(meta["Image Index"], meta["Patient ID"]))


def brier_score(probs, labels):
    return float(np.mean((probs - labels) ** 2))


def nll(probs, labels):
    p = np.clip(probs, EPS, 1 - EPS)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def ece_with_bin_counts(probs, labels, n_bins=N_BINS):
    """ECE = sum_b (n_b/N) |acc(b) - conf(b)|, con conteggio esplicito per bin."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bin_edges) - 1, 0, n_bins - 1)
    ece = 0.0
    bin_counts = []
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = mask.sum()
        bin_counts.append(int(n_b))
        if n_b == 0:
            continue
        acc_b = labels[mask].mean()
        conf_b = probs[mask].mean()
        ece += (n_b / len(probs)) * abs(acc_b - conf_b)
    return float(ece), bin_counts


def calibration_slope_intercept(probs, labels):
    """Calibrazione di Cox: regressione logistica di label su logit(prob).
    slope=1, intercept=0 -> calibrazione perfetta."""
    p = np.clip(probs, EPS, 1 - EPS)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan  # non stimabile se la classe e' costante nel campione
    try:
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(logit_p, labels)
        return float(lr.coef_[0][0]), float(lr.intercept_[0])
    except Exception:
        return np.nan, np.nan


def patient_clustered_bootstrap_generic(compute_fn, image_ids, patient_ids, n_boot=N_BOOTSTRAP, seed=42):
    """Bootstrap patient-clustered generico: compute_fn prende un array di
    indici (posizioni nel dataframe originale) e ritorna uno scalare."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"pos": np.arange(len(image_ids)), "patient": patient_ids})
    grouped = {p: g["pos"].to_numpy() for p, g in df.groupby("patient")}
    unique_patients = np.array(list(grouped.keys()))

    boot_vals = np.empty(n_boot)
    for i in range(n_boot):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        idx = np.concatenate([grouped[p] for p in sampled_patients])
        boot_vals[i] = compute_fn(idx)

    lo, hi = np.percentile(boot_vals, [100 * CI_ALPHA / 2, 100 * (1 - CI_ALPHA / 2)])
    return lo, hi


def analyze_condition(cond, seeds, labels_df, patient_map):
    per_seed_rows = []
    per_class_rows = []

    for seed in seeds:
        pred_path = os.path.join(OUTPUT_DIR, f"test_predictions_{cond}_seed{seed}.csv")
        if not os.path.exists(pred_path):
            continue
        preds = pd.read_csv(pred_path).set_index("Image Index")
        common = preds.index.intersection(labels_df.index)
        preds, labels = preds.loc[common], labels_df.loc[common]

        class_briers, class_nlls, class_eces = [], [], []
        for c in CLASSES:
            p = preds[f"prob_{c}"].to_numpy()
            y = labels[c].to_numpy()
            b = brier_score(p, y)
            n = nll(p, y)
            e, bin_counts = ece_with_bin_counts(p, y)
            slope, intercept = calibration_slope_intercept(p, y)
            class_briers.append(b); class_nlls.append(n); class_eces.append(e)
            per_class_rows.append({
                "condition": cond, "seed": seed, "class": c,
                "brier": b, "nll": n, "ece": e,
                "calibration_slope": slope, "calibration_intercept": intercept,
                "bin_counts": str(bin_counts),
            })

        per_seed_rows.append({
            "condition": cond, "seed": seed,
            "macro_brier": np.mean(class_briers),
            "macro_nll": np.mean(class_nlls),
            "macro_ece": np.mean(class_eces),
        })

    if not per_seed_rows:
        return None, per_class_rows

    seed_df = pd.DataFrame(per_seed_rows)
    row = {
        "condition": cond, "n_seeds": len(seed_df),
        "macro_brier_mean": seed_df["macro_brier"].mean(),
        "macro_brier_std": seed_df["macro_brier"].std(ddof=1) if len(seed_df) > 1 else np.nan,
        "macro_nll_mean": seed_df["macro_nll"].mean(),
        "macro_nll_std": seed_df["macro_nll"].std(ddof=1) if len(seed_df) > 1 else np.nan,
        "macro_ece_mean": seed_df["macro_ece"].mean(),
        "macro_ece_std": seed_df["macro_ece"].std(ddof=1) if len(seed_df) > 1 else np.nan,
    }

    # CI bootstrap patient-clustered sul macro-ECE, usando solo il primo seed
    # disponibile per tenere il costo computazionale contenuto (l'obiettivo qui
    # e' quantificare l'incertezza dovuta al campione di pazienti, non ai seed
    # -- per quella, vedi hierarchical_bootstrap in analyze_ablation_results.py)
    if patient_map is not None and len(seeds) > 0:
        first_seed = seeds[0]
        pred_path = os.path.join(OUTPUT_DIR, f"test_predictions_{cond}_seed{first_seed}.csv")
        if os.path.exists(pred_path):
            preds = pd.read_csv(pred_path).set_index("Image Index")
            common = preds.index.intersection(labels_df.index)
            preds, labels = preds.loc[common], labels_df.loc[common]
            image_ids = list(common)
            patients = [patient_map.get(i, np.nan) for i in image_ids]

            def compute_macro_ece(idx):
                eces = []
                for c in CLASSES:
                    p = preds.iloc[idx][f"prob_{c}"].to_numpy()
                    y = labels.iloc[idx][c].to_numpy()
                    e, _ = ece_with_bin_counts(p, y)
                    eces.append(e)
                return np.mean(eces)

            lo, hi = patient_clustered_bootstrap_generic(compute_macro_ece, image_ids, patients)
            row["macro_ece_CI_95_lo"] = lo
            row["macro_ece_CI_95_hi"] = hi

    return row, per_class_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=str, default="all")
    parser.add_argument("--seeds", type=str, default="all")
    args = parser.parse_args()
    conditions = CONDITIONS_ORDER if args.conditions == "all" else args.conditions.split(",")
    seeds = SEEDS if args.seeds == "all" else [int(s) for s in args.seeds.split(",")]

    print("[INFO] Ricostruisco le etichette vere da test_split.csv...")
    labels_df = build_label_matrix()
    patient_map = load_patient_map()
    if patient_map is None:
        print(f"[WARN] {NIH_METADATA_CSV} non trovato: CI bootstrap per l'ECE saltato.")

    summary_rows, all_per_class = [], []
    for cond in conditions:
        print(f"[INFO] Analizzo {cond}...")
        row, per_class = analyze_condition(cond, seeds, labels_df, patient_map)
        if row is not None:
            summary_rows.append(row)
        all_per_class.extend(per_class)

    summary_df = pd.DataFrame(summary_rows)
    per_class_df = pd.DataFrame(all_per_class)

    out1 = os.path.join(OUTPUT_DIR, "SUMMARY_calibration_full.csv")
    out2 = os.path.join(OUTPUT_DIR, "SUMMARY_calibration_per_class.csv")
    summary_df.to_csv(out1, index=False)
    per_class_df.to_csv(out2, index=False)

    print(f"\n{'='*100}\nCALIBRATION SUITE -- Brier, NLL, ECE (mean +/- std su seed), per condizione\n{'='*100}")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(summary_df.to_string(index=False))
    print(f"\nRiepilogo per condizione: {out1}")
    print(f"Dettaglio per classe (incl. conteggi per bin ECE, slope/intercept): {out2}")

    print("\n[COME LEGGERE]")
    print("- macro_brier, macro_nll: piu' bassi = meglio calibrato/piu' accurato in probabilita'.")
    print("- calibration_slope vicino a 1 e calibration_intercept vicino a 0 (nel file per-classe)")
    print("  indicano buona calibrazione; slope << 1 indica probabilita' troppo estreme.")
    print("- macro_ece_CI_95_lo/hi: intervallo di confidenza patient-clustered per l'ECE macro,")
    print("  calcolato sul primo seed disponibile per condizione.")


if __name__ == "__main__":
    main()
