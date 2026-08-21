"""
recompute_table7_per_class_ece.py

OBIETTIVO
---------
diagnose_ece_discrepancy.py ha gia' confermato che il checkpoint originale,
valutato con la pipeline nuova, da' ECE macro ~= 0.0106 (non piu' 0.1015).
Questo script prende le predizioni gia' salvate da quel diagnostico
(diagnostic_original_checkpoint_predictions.csv) e calcola il dettaglio
PER CLASSE, nello stesso formato della Table 7 del paper, cosi' possiamo
sostituire i 14 valori vecchi con quelli corretti e riconciliati.

Non serve GPU, non serve rilanciare l'inferenza: e' pura ri-analisi del
CSV gia' prodotto.

USO
---
    python recompute_table7_per_class_ece.py
"""

import numpy as np
import pandas as pd

PRED_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/ablation_results/diagnostic_original_checkpoint_predictions.csv"
TEST_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
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


def ece(probs, labels, n_bins=N_BINS):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bin_edges) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = mask.sum()
        if n_b == 0:
            continue
        e += (n_b / len(probs)) * abs(labels[mask].mean() - probs[mask].mean())
    return float(e)


def main():
    preds = pd.read_csv(PRED_CSV).set_index("Image Index")
    labels = build_label_matrix()
    common = preds.index.intersection(labels.index)
    preds, labels = preds.loc[common], labels.loc[common]

    rows = []
    for c in CLASSES:
        p = preds[f"prob_{c}"].to_numpy()
        y = labels[c].to_numpy()
        rows.append({"Pathology": c, "ECE": round(ece(p, y), 4)})

    result = pd.DataFrame(rows).sort_values("ECE").reset_index(drop=True)
    macro = result["ECE"].mean()

    print("=" * 60)
    print("TABLE 7 CORRETTA -- ECE per classe, checkpoint originale, pipeline nuova")
    print("=" * 60)
    for _, r in result.iterrows():
        print(f"  {r['Pathology']:20s} {r['ECE']:.4f}")
    print(f"\n  {'Mean (macro)':20s} {macro:.4f}")
    print("\n[NEXT STEP] Sostituisci i valori della Table 7 nel paper con questi,")
    print("e il valore 0.1015 con il nuovo macro riportato sopra.")

    result.to_csv("/home/gpuvm/Desktop/Luca Migliaccio/ablation_results/table7_corrected.csv", index=False)


if __name__ == "__main__":
    main()
