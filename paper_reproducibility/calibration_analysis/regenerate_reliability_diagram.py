"""
regenerate_reliability_diagram.py

OBIETTIVO
---------
La Figura 7 del paper (reliability diagram per la classe Effusion) mostrava
ECE=0.0530, calcolato con la vecchia pipeline di valutazione ormai
superata. Un revisore ha correttamente notato che tenere una figura con un
numero dichiaratamente sbagliato non e' accettabile, e che non si puo'
presumere che solo il numero sia cambiato mentre la forma della curva resta
uguale -- va rigenerata dai dati corretti, non semplicemente ri-etichettata.

Questo script usa le predizioni gia' salvate da diagnose_ece_discrepancy.py
(diagnostic_original_checkpoint_predictions.csv, calcolate con la pipeline
di valutazione corretta e unificata) per rigenerare il reliability diagram
da zero: assi ed etichette in inglese, e i conteggi campione per bin
mostrati esplicitamente sotto ogni punto (un'altra richiesta della
revisione, "with bin counts").

Non serve GPU, non serve rilanciare l'inferenza -- e' pura ri-analisi del
CSV gia' prodotto.

USO
---
    python regenerate_reliability_diagram.py
    python regenerate_reliability_diagram.py --pathology Atelectasis

Di default rigenera il diagramma per Effusion (la stessa classe della
Figura 7 originale, per continuita'); puoi generarne altre con --pathology.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRED_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/ablation_results/diagnostic_original_checkpoint_predictions.csv"
TEST_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
OUTPUT_PNG = "/home/gpuvm/Desktop/Luca Migliaccio/ablation_results/fig7_reliability_diagram_CORRECTED.png"
N_BINS = 10

CLASSES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
           "Fibrosis", "Pleural_Thickening", "Hernia"]


def build_label_matrix():
    df = pd.read_csv(TEST_CSV)
    labels = pd.DataFrame(0, index=df["Image Index"], columns=CLASSES)
    for _, row in df.iterrows():
        for l in str(row["Finding Labels"]).split("|"):
            if l in CLASSES:
                labels.loc[row["Image Index"], l] = 1
    return labels


def reliability_curve(probs, labels, n_bins=N_BINS):
    """Ritorna, per ciascun bin: confidenza media predetta, accuratezza
    empirica osservata, e conteggio campioni -- tutto esplicito, cosi' i
    bin con pochissimi campioni (statisticamente rumorosi) sono visibili
    invece che nascosti."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bin_edges) - 1, 0, n_bins - 1)
    mean_conf, mean_acc, counts = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = mask.sum()
        counts.append(int(n_b))
        if n_b == 0:
            mean_conf.append(np.nan)
            mean_acc.append(np.nan)
        else:
            mean_conf.append(probs[mask].mean())
            mean_acc.append(labels[mask].mean())
    return np.array(mean_conf), np.array(mean_acc), np.array(counts)


def ece_from_curve(mean_conf, mean_acc, counts):
    n_total = counts.sum()
    valid = counts > 0
    return float(np.sum(counts[valid] / n_total * np.abs(mean_acc[valid] - mean_conf[valid])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathology", type=str, default="Effusion")
    args = parser.parse_args()
    pathology = args.pathology
    if pathology not in CLASSES:
        raise ValueError(f"Classe sconosciuta: {pathology}. Scegli tra: {CLASSES}")

    preds = pd.read_csv(PRED_CSV).set_index("Image Index")
    labels = build_label_matrix()
    common = preds.index.intersection(labels.index)
    preds, labels = preds.loc[common], labels.loc[common]

    p = preds[f"prob_{pathology}"].to_numpy()
    y = labels[pathology].to_numpy()
    mean_conf, mean_acc, counts = reliability_curve(p, y)
    ece = ece_from_curve(mean_conf, mean_acc, counts)

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=300)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    valid = counts > 0
    ax.plot(mean_conf[valid], mean_acc[valid], marker="o", linewidth=2,
            color="#1f7a6f", label=f"ConvNeXt ({pathology})")

    # conteggio campioni per bin mostrato esplicitamente accanto a ogni punto
    for x, yv, n in zip(mean_conf[valid], mean_acc[valid], counts[valid]):
        ax.annotate(f"n={n}", (x, yv), textcoords="offset points", xytext=(6, -10), fontsize=8, color="dimgray")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed fraction of positives (accuracy)")
    ax.set_title(f"Reliability Diagram -- {pathology}\n(ECE: {ece:.4f}, corrected evaluation pipeline)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_PNG.replace("CORRECTED", f"CORRECTED_{pathology}")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"[DONE] {out_path}")
    print(f"Corrected ECE for {pathology}: {ece:.4f}")
    print("Per-bin sample counts:", dict(zip(range(N_BINS), counts)))
    print("\n[NEXT STEP] Replace figures/fig7_reliability_diagram.png with this file,")
    print("and update the ECE value in the figure caption in main.tex accordingly.")


if __name__ == "__main__":
    main()
