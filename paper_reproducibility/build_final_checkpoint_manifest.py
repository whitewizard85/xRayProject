"""
build_final_checkpoint_manifest.py

OBIETTIVO
---------
Prende l'inventario completo (checkpoint_inventory_FULL.csv, gia' generato
da inventory_checkpoints.py) e produce il manifest FINALE E PULITO da
mettere nel repository di riproducibilita': solo i checkpoint davvero
citati nel paper, con una whitelist esplicita (non un glob che prende
tutto), cosi' non c'e' ambiguita' su quali dei 40+ file in checkpoints/
siano effettivamente rilevanti.

Include anche una riga esplicita per ciascuno dei DUE checkpoint mancanti
(ResNet-50 ImageNet puro, DenseNet-121 ImageNet puro): questi modelli sono
stati allenati dallo studente sul proprio PC locale, PRIMA di spostare il
lavoro sul server GPU, e i relativi checkpoint non sono mai stati caricati.
Li documentiamo esplicitamente come "NOT AVAILABLE" nel manifest invece di
ometterli silenziosamente: chi legge il repository deve sapere che quei due
risultati di Tabella 4 non sono verificabili a partire da un checkpoint,
solo dalle metriche gia' riportate nel paper.

USO
---
    python build_final_checkpoint_manifest.py

Richiede che results/checkpoint_inventory_FULL.csv esista gia' (prodotto da
inventory_checkpoints.py).
"""

import os
import pandas as pd

INVENTORY_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/paper_reproducibility/results/checkpoint_inventory_FULL.csv"
OUTPUT_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/paper_reproducibility/results/checkpoint_hashes.csv"

# whitelist esplicita: nome file -> (etichetta usata nel paper, cartella di origine)
# NESSUN glob, NESSUN pattern -- solo questi file, deliberatamente.
WHITELIST = {
    # --- Table 4: single-backbone / ensemble comparison ---
    "best_convnext_base_22k.pth":     "Table 4 -- ConvNeXt, ImageNet-22k (pre-intervention / reference for ablation C0)",
    "best_debiased_convnext.pth":     "Table 4/7 -- ConvNeXt post-intervention (original single-instance reference, superseded by ablation C6 seeds)",
    "best_densenet121_v4_xrv.pth":    "Table 4 -- DenseNet-121, TorchXRayVision",
    "best_resnet50_v5_xrv.pth":       "Table 4 -- ResNet-50, TorchXRayVision",
    "best_efficientnet_b7_asl.pth":   "Table 4 -- EfficientNet-B7, ImageNet-22k",
    "best_swin_v2.pth":               "Table 4 -- SwinV2, ImageNet-22k",
    "best_swin_biomedical_v2.pth":    "Table 4 -- SwinV2, radiology-pretrained",

    # --- Ablation (Tables 5, 6, 8): C0-C6 x seed 42/123/2026 ---
    "C0_baseline_seed42.pth":         "Ablation -- C0 baseline, seed 42",
    "C0_baseline_seed123.pth":        "Ablation -- C0 baseline, seed 123",
    "C0_baseline_seed2026.pth":       "Ablation -- C0 baseline, seed 2026",
    "C1_crop_only_seed42.pth":        "Ablation -- C1 crop only, seed 42",
    "C1_crop_only_seed123.pth":       "Ablation -- C1 crop only, seed 123",
    "C1_crop_only_seed2026.pth":      "Ablation -- C1 crop only, seed 2026",
    "C2_bce_only_seed42.pth":         "Ablation -- C2 loss only, seed 42",
    "C2_bce_only_seed123.pth":        "Ablation -- C2 loss only, seed 123",
    "C2_bce_only_seed2026.pth":       "Ablation -- C2 loss only, seed 2026",
    "C3_schedule_only_seed42.pth":    "Ablation -- C3 schedule only, seed 42",
    "C3_schedule_only_seed123.pth":   "Ablation -- C3 schedule only, seed 123",
    "C3_schedule_only_seed2026.pth":  "Ablation -- C3 schedule only, seed 2026",
    "C4_noaug_only_seed42.pth":       "Ablation -- C4 augmentation removed, seed 42",
    "C4_noaug_only_seed123.pth":      "Ablation -- C4 augmentation removed, seed 123",
    "C4_noaug_only_seed2026.pth":     "Ablation -- C4 augmentation removed, seed 2026",
    "C5_crop_bce_seed42.pth":         "Ablation -- C5 crop + loss, seed 42",
    "C5_crop_bce_seed123.pth":        "Ablation -- C5 crop + loss, seed 123",
    "C5_crop_bce_seed2026.pth":       "Ablation -- C5 crop + loss, seed 2026",
    "C6_full_combined_seed42.pth":    "Ablation -- C6 full combination, seed 42",
    "C6_full_combined_seed123.pth":   "Ablation -- C6 full combination, seed 123",
    "C6_full_combined_seed2026.pth":  "Ablation -- C6 full combination, seed 2026",
}

# Checkpoint di Tabella 4 esistenti nel paper ma il cui file non e' mai
# stato caricato sul server (allenati localmente dallo studente prima del
# passaggio al server GPU). Documentati esplicitamente, non omessi.
MISSING_CHECKPOINTS = [
    {
        "source_folder": "NOT_AVAILABLE",
        "filename": "(never uploaded)",
        "size_MB": "",
        "last_modified": "",
        "sha256": "",
        "used_in_paper": "Table 4 -- ResNet-50, ImageNet (pure) -- trained locally by the "
                          "student before migrating to the GPU server; checkpoint was never "
                          "uploaded and is not recoverable. Only the metrics reported in "
                          "Table 4 are verifiable for this configuration, not the model itself.",
    },
    {
        "source_folder": "NOT_AVAILABLE",
        "filename": "(never uploaded)",
        "size_MB": "",
        "last_modified": "",
        "sha256": "",
        "used_in_paper": "Table 4 -- DenseNet-121, ImageNet (pure) -- trained locally by the "
                          "student before migrating to the GPU server; checkpoint was never "
                          "uploaded and is not recoverable. Only the metrics reported in "
                          "Table 4 are verifiable for this configuration, not the model itself.",
    },
]


def main():
    if not os.path.exists(INVENTORY_CSV):
        raise FileNotFoundError(
            f"{INVENTORY_CSV} non trovato -- lancia prima inventory_checkpoints.py")

    inv = pd.read_csv(INVENTORY_CSV)
    inv["filename_only"] = inv["filename"]

    kept_rows = []
    found_filenames = set()
    for _, row in inv.iterrows():
        fname = row["filename_only"]
        if fname in WHITELIST:
            row = row.copy()
            row["used_in_paper"] = WHITELIST[fname]
            kept_rows.append(row)
            found_filenames.add(fname)

    missing_from_whitelist = set(WHITELIST.keys()) - found_filenames
    if missing_from_whitelist:
        print("[ATTENZIONE] Questi file della whitelist non sono stati trovati nell'inventario:")
        for f in sorted(missing_from_whitelist):
            print(f"  - {f}")
        print("Controlla i nomi esatti in checkpoint_inventory_FULL.csv prima di procedere.\n")

    final_df = pd.DataFrame(kept_rows)[
        ["source_folder", "filename", "size_MB", "last_modified", "sha256", "used_in_paper"]
    ]
    missing_df = pd.DataFrame(MISSING_CHECKPOINTS)
    final_df = pd.concat([final_df, missing_df], ignore_index=True)

    final_df.to_csv(OUTPUT_CSV, index=False)

    n_available = len(kept_rows)
    n_missing = len(MISSING_CHECKPOINTS)
    print(f"[DONE] {OUTPUT_CSV}")
    print(f"  {n_available} checkpoint disponibili e hashati")
    print(f"  {n_missing} checkpoint mancanti (documentati esplicitamente, non ricostruibili)")
    print(f"  Totale righe nel manifest: {n_available + n_missing}")
    print(f"\n[EXPECTED] 7 (Table 4 single-backbone/ensemble) + 21 (ablation) = 28 checkpoint")
    print(f"           attesi in totale, ma 2 (ResNet-50/DenseNet-121 ImageNet puro) mancanti")
    print(f"           => 26 disponibili + 2 documentati come mancanti = 28 righe totali.")
    if n_available != 26:
        print(f"\n[ATTENZIONE] Trovati {n_available} checkpoint disponibili, attesi 26 -- controlla sopra.")


if __name__ == "__main__":
    main()
