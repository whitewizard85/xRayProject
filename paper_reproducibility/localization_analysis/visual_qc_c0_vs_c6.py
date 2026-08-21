"""
visual_qc_c0_vs_c6.py

OBIETTIVO
---------
Il risultato quantitativo dell'ablation dice che C6_full_combined ha
un'attivazione Grad-CAM MENO concentrata dentro la sagoma corporea rispetto
a C0_baseline (effetto opposto a quanto suggeriva il confronto originale
non controllato). Prima di riscrivere il paper su questo risultato, questo
script genera un controllo visivo: prende le immagini con lo spostamento
piu' estremo (sia peggiorativo che migliorativo) e mostra fianco a fianco
originale / maschera corporea / Grad-CAM C0 / Grad-CAM C6, cosi' puoi
vedere con i tuoi occhi dove va davvero l'attivazione, non solo il numero
aggregato.

Usa lo stesso preprocessing condiviso (420->384) e lo stesso identificativo
di classe Top-1 (dal modello C6) gia' usati nella metrica quantitativa, per
coerenza totale con i numeri che hai gia' ottenuto.

PREREQUISITI
------------
Richiede che tu abbia gia' completato (almeno un seed di) C0_baseline e
C6_full_combined con ablation_train_and_eval.py, incluse le rispettive
localization_C0_baseline_seed<S>.csv e localization_C6_full_combined_seed<S>.csv.

USO
---
    python visual_qc_c0_vs_c6.py
    python visual_qc_c0_vs_c6.py --seed 123          # se vuoi un altro seed
    python visual_qc_c0_vs_c6.py --n_worst 5 --n_best 3 --n_median 3

OUTPUT
------
Una singola immagine PNG con una griglia di pannelli, salvata in
ablation_results/visual_qc/qc_grid.png, piu' un CSV con i valori esatti
delle immagini selezionate (visual_qc_selected_images.csv).
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from scipy import ndimage
import timm
from pytorch_grad_cam import GradCAM

# ============================================================
# CONFIGURAZIONE -- stessa della pipeline principale
# ============================================================
ROOT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
CHECKPOINT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/ablation_checkpoints"
OUTPUT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/ablation_results"
QC_DIR = os.path.join(OUTPUT_DIR, "visual_qc")
os.makedirs(QC_DIR, exist_ok=True)

IMAGE_SIZE = 384
CROP_RESIZE = 420
CLASSES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
           "Fibrosis", "Pleural_Thickening", "Hernia"]
NUM_CLASSES = len(CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image_path(img_name):
    for i in range(1, 13):
        path = os.path.join(ROOT_DIR, f"images_{i:03d}", "images", img_name)
        if os.path.exists(path):
            return path
    return None


def load_model(ckpt_path):
    model = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    return model


def body_silhouette_mask(pil_img_gray, erosion_px=6):
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_erosion, disk
    arr = np.array(pil_img_gray.convert("L"), dtype=np.float64)
    mask = arr > threshold_otsu(arr)
    labeled, n = ndimage.label(mask)
    if n > 0:
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        mask = labeled == (np.argmax(sizes) + 1)
    mask = ndimage.binary_fill_holes(mask)
    if erosion_px > 0:
        mask = binary_erosion(mask, footprint=disk(erosion_px))
    return mask.astype(np.float32)


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        return model_output[:, self.category] if model_output.ndim > 1 else model_output[self.category]


def select_images(seed, n_worst, n_best, n_median):
    """Sceglie le immagini con lo spostamento (C6 - C0) piu' estremo in entrambe le
    direzioni, piu' alcune vicine alla mediana, per un quadro rappresentativo."""
    c0_path = os.path.join(OUTPUT_DIR, f"localization_C0_baseline_seed{seed}.csv")
    c6_path = os.path.join(OUTPUT_DIR, f"localization_C6_full_combined_seed{seed}.csv")
    if not (os.path.exists(c0_path) and os.path.exists(c6_path)):
        raise FileNotFoundError(
            f"Servono entrambi:\n  {c0_path}\n  {c6_path}\n"
            f"Completa C0_baseline e C6_full_combined per il seed {seed} prima di lanciare questo script."
        )
    c0 = pd.read_csv(c0_path)[["Image Index", "fraction_inside"]].rename(columns={"fraction_inside": "frac_C0"})
    c6 = pd.read_csv(c6_path)[["Image Index", "fraction_inside"]].rename(columns={"fraction_inside": "frac_C6"})
    merged = pd.merge(c0, c6, on="Image Index").dropna()
    merged["diff"] = merged["frac_C6"] - merged["frac_C0"]
    merged = merged.sort_values("diff").reset_index(drop=True)

    worst = merged.head(n_worst)  # C6 molto peggio di C0 (diff piu' negativo)
    best = merged.tail(n_best)    # C6 molto meglio di C0 (diff piu' positivo)
    mid_start = len(merged) // 2 - n_median // 2
    median = merged.iloc[mid_start: mid_start + n_median]

    selected = pd.concat([
        worst.assign(group="worst_for_C6"),
        median.assign(group="near_median"),
        best.assign(group="best_for_C6"),
    ]).reset_index(drop=True)
    return selected


def make_grid(selected_df, seed, model_c0, model_c6):
    shared_tf = transforms.Compose([
        transforms.Resize((CROP_RESIZE, CROP_RESIZE)), transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_only_tf = transforms.Compose([
        transforms.Resize((CROP_RESIZE, CROP_RESIZE)), transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE))
    ])

    cam_c0 = GradCAM(model=model_c0, target_layers=[model_c0.stages[-1]])
    cam_c6 = GradCAM(model=model_c6, target_layers=[model_c6.stages[-1]])

    n = len(selected_df)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Original", "Body mask", "Grad-CAM C0_baseline", "Grad-CAM C6_full_combined"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=12, fontweight="bold")

    for i, row in selected_df.iterrows():
        img_path = get_image_path(row["Image Index"])
        img_pil = Image.open(img_path).convert("RGB")
        input_tensor = shared_tf(img_pil).unsqueeze(0).to(device)
        img_np = np.array(img_only_tf(img_pil)).astype(float) / 255.0
        mask = body_silhouette_mask(img_only_tf(img_pil))

        with torch.no_grad():
            probs_c6 = torch.sigmoid(model_c6(input_tensor)).cpu().numpy()[0]
        top1 = int(np.argmax(probs_c6))
        target = [ClassifierOutputTarget(top1)]

        gcam_c0 = cam_c0(input_tensor=input_tensor, targets=target)[0, :]
        gcam_c6 = cam_c6(input_tensor=input_tensor, targets=target)[0, :]

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_ylabel(f"{row['group']}\n{row['Image Index']}\n"
                               f"class={CLASSES[top1]}\nC0={row['frac_C0']:.3f} C6={row['frac_C6']:.3f}",
                               fontsize=8, rotation=0, ha="right", va="center", labelpad=60)
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 2].imshow(img_np); axes[i, 2].imshow(gcam_c0, cmap="jet", alpha=0.5)
        axes[i, 3].imshow(img_np); axes[i, 3].imshow(gcam_c6, cmap="jet", alpha=0.5)
        for j in range(4):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(QC_DIR, f"qc_grid_seed{seed}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_worst", type=int, default=4, help="immagini dove C6 peggiora di piu' vs C0")
    parser.add_argument("--n_best", type=int, default=4, help="immagini dove C6 migliora di piu' vs C0")
    parser.add_argument("--n_median", type=int, default=4, help="immagini vicine alla mediana della differenza")
    args = parser.parse_args()

    selected = select_images(args.seed, args.n_worst, args.n_best, args.n_median)
    selected.to_csv(os.path.join(QC_DIR, f"visual_qc_selected_images_seed{args.seed}.csv"), index=False)
    print(f"[INFO] Immagini selezionate:\n{selected.to_string(index=False)}")

    ckpt_c0 = os.path.join(CHECKPOINT_DIR, f"C0_baseline_seed{args.seed}.pth")
    ckpt_c6 = os.path.join(CHECKPOINT_DIR, f"C6_full_combined_seed{args.seed}.pth")
    model_c0 = load_model(ckpt_c0)
    model_c6 = load_model(ckpt_c6)

    make_grid(selected, args.seed, model_c0, model_c6)
    print("\n[NEXT STEP] Apri qc_grid_seed<S>.png e guarda le righe 'worst_for_C6':")
    print("  se l'attivazione C6 si sposta su marker/testo/angoli/bordo dell'immagine,")
    print("  la maschera corporea sta facendo il suo lavoro correttamente e il")
    print("  risultato quantitativo e' probabilmente affidabile (solo controintuitivo).")
    print("  Se invece C6 si sposta su regioni interne plausibili (cuore, mediastino,")
    print("  costole) che semplicemente non sono polmone, il problema e' che la")
    print("  maschera 'corpo' non e' una maschera 'polmone' -- coerente con il limite")
    print("  gia' dichiarato nel paper, ma va detto esplicitamente nella riscrittura.")


if __name__ == "__main__":
    main()
