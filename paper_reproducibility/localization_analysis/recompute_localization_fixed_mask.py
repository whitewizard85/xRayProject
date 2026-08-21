"""
recompute_localization_fixed_mask.py

OBIETTIVO
---------
Il controllo visivo (qc_grid_seed42.png) ha mostrato un difetto specifico
nella maschera "sagoma corporea" usata finora: in diversi casi la spalla
del paziente viene esclusa dal corpo, perche' alla giuntura della spalla
l'intensita' del raggio X scende abbastanza da spezzare in due blob
separati (busto + spalla) la sogliatura di Otsu, e il codice originale
teneva solo il blob piu' grande (il busto), scartando la spalla.

Questo script corregge la maschera aggiungendo una CHIUSURA MORFOLOGICA
(dilatazione seguita da erosione con lo stesso elemento strutturante,
raggio configurabile) PRIMA di cercare la componente connessa piu' grande:
questo salda i piccoli varchi alla giuntura della spalla, cosi' spalla e
busto restano un unico blob e vengono inclusi insieme.

Ricalcola poi la metrica di localizzazione (stesso campione di 1000
immagini, stesso seed=42, stessa procedura) per C0_baseline e
C6_full_combined con la maschera corretta, cosi' puoi confrontare il
risultato "prima" (maschera con difetto) e "dopo" (maschera corretta) e
vedere quanto del segnale controintuitivo sopravvive.

USO
---
    python recompute_localization_fixed_mask.py
    python recompute_localization_fixed_mask.py --conditions C0_baseline,C6_full_combined --seeds 42

OUTPUT
------
  - ablation_results/localization_<condition>_seed<seed>_FIXEDMASK.csv
  - ablation_results/mask_comparison/  (6 immagini di controllo: vecchia
    maschera vs nuova maschera, sulle stesse immagini "worst_for_C6" gia'
    identificate, per verificare visivamente che la spalla ora sia inclusa)
  - stampa a schermo un confronto rapido vecchia vs nuova metrica
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
from tqdm import tqdm
from pytorch_grad_cam import GradCAM

# ============================================================
# CONFIGURAZIONE
# ============================================================
ROOT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
CHECKPOINT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/ablation_checkpoints"
TEST_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
OUTPUT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/ablation_results"
MASK_QC_DIR = os.path.join(OUTPUT_DIR, "mask_comparison")
os.makedirs(MASK_QC_DIR, exist_ok=True)

IMAGE_SIZE = 384
CROP_RESIZE = 420
N_SAMPLES = 1000
SAMPLE_SEED = 42  # stesso campione di sempre, per confrontabilita' diretta

CLASSES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
           "Fibrosis", "Pleural_Thickening", "Hernia"]
NUM_CLASSES = len(CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLOSING_RADIUS_PX = 20  # raggio della chiusura morfologica; 20px su 384 e' abbastanza
                         # da saldare il varco alla spalla senza fondere strutture distanti


def get_image_path(img_name):
    for i in range(1, 13):
        path = os.path.join(ROOT_DIR, f"images_{i:03d}", "images", img_name)
        if os.path.exists(path):
            return path
    return None


def body_silhouette_mask_OLD(pil_img_gray, erosion_px=6):
    """Versione originale (con il difetto della spalla), tenuta qui solo per il confronto visivo."""
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


def body_silhouette_mask_FIXED(pil_img_gray, closing_radius=CLOSING_RADIUS_PX, erosion_px=6):
    """Versione corretta: chiusura morfologica prima di isolare la componente
    connessa piu' grande, per saldare il varco busto-spalla."""
    from skimage.filters import threshold_otsu
    from skimage.morphology import erosion, closing, disk
    arr = np.array(pil_img_gray.convert("L"), dtype=np.float64)
    mask = arr > threshold_otsu(arr)

    # *** LA CORREZIONE ***: chiude i piccoli varchi (es. giuntura spalla-busto)
    # PRIMA di cercare la componente connessa piu' grande, cosi' spalla e
    # busto restano attaccati invece di essere trattati come blob separati.
    mask = closing(mask, footprint=disk(closing_radius))

    labeled, n = ndimage.label(mask)
    if n > 0:
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        mask = labeled == (np.argmax(sizes) + 1)
    mask = ndimage.binary_fill_holes(mask)
    if erosion_px > 0:
        mask = erosion(mask, footprint=disk(erosion_px))
    return mask.astype(np.float32)


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        return model_output[:, self.category] if model_output.ndim > 1 else model_output[self.category]


def load_model(ckpt_path):
    model = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    return model


def save_mask_comparison_qc(sample_images):
    """Confronto visivo maschera vecchia vs nuova sulle stesse immagini
    'worst_for_C6' gia' identificate nel controllo precedente, per
    verificare che la correzione includa davvero la spalla."""
    img_tf = transforms.Compose([
        transforms.Resize((CROP_RESIZE, CROP_RESIZE)), transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE))
    ])
    n = len(sample_images)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    col_titles = ["Original", "OLD mask (bug)", "FIXED mask (shoulder closed)"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=11, fontweight="bold")

    for i, img_name in enumerate(sample_images):
        path = get_image_path(img_name)
        if path is None:
            continue
        img_pil = img_tf(Image.open(path).convert("RGB"))
        img_np = np.array(img_pil).astype(float) / 255.0
        old_mask = body_silhouette_mask_OLD(img_pil)
        new_mask = body_silhouette_mask_FIXED(img_pil)

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_ylabel(img_name, fontsize=8, rotation=0, ha="right", va="center", labelpad=50)
        axes[i, 1].imshow(old_mask, cmap="gray")
        axes[i, 2].imshow(new_mask, cmap="gray")
        for j in range(3):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(MASK_QC_DIR, "mask_old_vs_fixed.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[SAVED] {out_path}  -- controlla che la spalla sia ora bianca (inclusa) nella colonna 3")


def run_condition(condition_name, seed):
    tag = f"{condition_name}_seed{seed}"
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{tag}.pth")
    out_csv = os.path.join(OUTPUT_DIR, f"localization_{tag}_FIXEDMASK.csv")
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] checkpoint mancante per {tag}")
        return
    if os.path.exists(out_csv):
        print(f"[SKIP] {out_csv} gia' presente")
        return

    model = load_model(ckpt_path)
    cam = GradCAM(model=model, target_layers=[model.stages[-1]])

    shared_tf = transforms.Compose([
        transforms.Resize((CROP_RESIZE, CROP_RESIZE)), transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_only_tf = transforms.Compose([
        transforms.Resize((CROP_RESIZE, CROP_RESIZE)), transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE))
    ])

    df = pd.read_csv(TEST_CSV)
    n = min(N_SAMPLES, len(df))
    sample_df = df.sample(n=n, random_state=SAMPLE_SEED).reset_index(drop=True)

    rows = []
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc=f"FixedMask {tag}"):
        img_path = get_image_path(row["Image Index"])
        if img_path is None:
            continue
        img_pil = Image.open(img_path).convert("RGB")
        input_tensor = shared_tf(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(input_tensor)).cpu().numpy()[0]
        top1 = int(np.argmax(probs))
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(top1)])[0, :]
        mask = body_silhouette_mask_FIXED(img_only_tf(img_pil))
        cam_pos = np.clip(grayscale_cam, 0, None)
        total = cam_pos.sum()
        frac_inside = float((cam_pos * mask).sum() / total) if total > 1e-8 else np.nan
        rows.append({"Image Index": row["Image Index"], "condition": condition_name, "seed": seed,
                      "top1_class": CLASSES[top1], "fraction_inside": frac_inside})

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[DONE] {out_csv}")


def print_before_after(condition_name, seed):
    tag = f"{condition_name}_seed{seed}"
    old_path = os.path.join(OUTPUT_DIR, f"localization_{tag}.csv")
    new_path = os.path.join(OUTPUT_DIR, f"localization_{tag}_FIXEDMASK.csv")
    if not (os.path.exists(old_path) and os.path.exists(new_path)):
        return
    old = pd.read_csv(old_path)["fraction_inside"].mean()
    new = pd.read_csv(new_path)["fraction_inside"].mean()
    print(f"  {tag:35s}  OLD mask mean={old:.4f}   FIXED mask mean={new:.4f}   delta={new-old:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=str, default="C0_baseline,C6_full_combined")
    parser.add_argument("--seeds", type=str, default="42")
    args = parser.parse_args()
    conditions = args.conditions.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    # 1) confronto visivo maschera vecchia/nuova sulle immagini worst_for_C6 gia' note
    qc_csv = os.path.join(OUTPUT_DIR, "visual_qc", "visual_qc_selected_images_seed42.csv")
    if os.path.exists(qc_csv):
        worst_images = pd.read_csv(qc_csv)
        worst_images = worst_images[worst_images["group"] == "worst_for_C6"]["Image Index"].tolist()
        save_mask_comparison_qc(worst_images)
    else:
        print("[INFO] visual_qc_selected_images_seed42.csv non trovato, salto il confronto maschere.")

    # 2) ricalcolo della metrica di localizzazione con la maschera corretta
    for cond in conditions:
        for seed in seeds:
            run_condition(cond, seed)

    # 3) confronto rapido vecchio vs nuovo, per condizione
    print(f"\n{'='*90}\nCONFRONTO MEDIA: maschera vecchia (con bug) vs maschera corretta\n{'='*90}")
    for cond in conditions:
        for seed in seeds:
            print_before_after(cond, seed)

    print("\n[NEXT STEP] Rilancia analyze_ablation_results.py puntando ai file *_FIXEDMASK.csv")
    print("(o aggiorna OUTPUT_DIR/nomi file in quello script) per vedere se C6 vs C0 resta")
    print("negativo, si annulla, o si inverte con la maschera corretta.")


if __name__ == "__main__":
    main()
