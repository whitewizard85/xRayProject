"""
bbox_pointing_game_localization.py

OBIETTIVO
---------
La critica piu' seria della revisione: la metrica "frazione di Grad-CAM
dentro la sagoma corporea" non misura la localizzazione della patologia,
solo la presenza di attivazione dentro un contorno molto largo (che include
cuore, coste, colonna, tessuti molli). Il dataset NIH ChestX-ray14 include
pero' un file ufficiale, BBox_List_2017.csv, con le coordinate reali di
~1000 lesioni annotate manualmente per 8 classi patologiche (lo stesso file
gia' usato per generare la Figura 1 del paper). Questo script lo sfrutta
per calcolare una metrica di localizzazione vera, contro ground truth
anatomico/patologico reale, non contro un proxy.

Per ogni immagine con bounding box nota, calcola la Grad-CAM per la CLASSE
VERA della lesione annotata (non il top-1 predetto: qui abbiamo la ground
truth, quindi ha senso usarla), poi calcola due metriche standard:

  1. POINTING-GAME ACCURACY: il pixel di massima attivazione della Grad-CAM
     cade dentro la bounding box? (1 = si, 0 = no). Metrica standard e
     conservativa nella letteratura di weakly-supervised localization.
  2. ENERGY-IN-BBOX: frazione della massa totale di attivazione Grad-CAM
     che cade dentro la bounding box (stessa logica della metrica
     "in-body" gia' nel paper, ma qui contro un riquadro patologico reale
     invece che contro la sagoma corporea).

NON serve nuovo training: riusa i checkpoint gia' allenati
(ablation_checkpoints/<condizione>_seed<seed>.pth). E' pero' inferenza GPU
vera (forward + Grad-CAM), quindi piu' lenta della pura ri-analisi CSV ma
molto piu' veloce di un training: qualche minuto per condizione su ~880-1000
immagini.

USO
---
    python bbox_pointing_game_localization.py
    python bbox_pointing_game_localization.py --conditions C0_baseline,C6_full_combined --seeds 42

Di default gira solo su C0_baseline e C6_full_combined (il confronto
principale del paper) per contenere i tempi; estendi con --conditions se
vuoi anche le altre condizioni dell'ablation.

OUTPUT
------
  - ablation_results/bbox_localization_<condizione>_seed<seed>.csv
    (una riga per immagine annotata: pointing_game_hit, energy_in_bbox)
  - ablation_results/SUMMARY_bbox_localization.csv (aggregato per
    condizione: accuracy media del pointing-game, energy-in-bbox media,
    con intervallo di confidenza bootstrap)
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import timm
from tqdm import tqdm
from pytorch_grad_cam import GradCAM

# ============================================================
# CONFIGURAZIONE
# ============================================================
ROOT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
BBOX_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/archive/BBox_List_2017.csv"
TEST_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
CHECKPOINT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/ablation_checkpoints"
OUTPUT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/ablation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 384
CROP_RESIZE = 420
N_BOOTSTRAP = 5000
CI_ALPHA = 0.05

# BBox_List_2017.csv usa nomi di classe leggermente diversi da Data_Entry_2017.csv
# per alcune patologie (es. "Infiltrate" invece di "Infiltration") -- mappatura
# esplicita per evitare mismatch silenziosi.
BBOX_LABEL_TO_CLASS = {
    "Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly", "Effusion": "Effusion",
    "Infiltrate": "Infiltration", "Infiltration": "Infiltration", "Mass": "Mass",
    "Nodule": "Nodule", "Pneumonia": "Pneumonia", "Pneumothorax": "Pneumothorax",
}
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


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        return model_output[:, self.category] if model_output.ndim > 1 else model_output[self.category]


def load_bbox_annotations():
    """Carica BBox_List_2017.csv, lo filtra SOLO sulle immagini del test split
    (fondamentale: BBox_List_2017.csv copre l'intero dataset NIH, incluse
    immagini che potrebbero essere state usate in training/validation; senza
    questo filtro il pointing-game sarebbe in parte una valutazione in-sample,
    non una vera misura di generalizzazione), e mappa ogni riga sulla nostra
    tassonomia di classi. Stampa esplicitamente sia il conteggio delle
    ANNOTAZIONI (righe, un'immagine puo' averne piu' di una se ha piu'
    patologie annotate) sia il conteggio delle IMMAGINI UNICHE, per evitare
    di confonderli come e' successo in una versione precedente di questa
    analisi."""
    df = pd.read_csv(BBOX_CSV)
    # Il file ufficiale NIH ha colonne: Image Index, Finding Label, Bbox [x, y, w, h]
    # (i nomi esatti delle colonne coordinate possono variare leggermente a seconda
    # della versione del CSV -- adatta qui se il tuo file ha intestazioni diverse)
    df.columns = [c.strip() for c in df.columns]
    coord_cols = [c for c in df.columns if c.lower() not in ("image index", "finding label")][:4]
    df = df.rename(columns={
        "Finding Label": "label",
        coord_cols[0]: "x", coord_cols[1]: "y", coord_cols[2]: "w", coord_cols[3]: "h",
    })
    df["mapped_class"] = df["label"].str.strip().map(BBOX_LABEL_TO_CLASS)
    df = df.dropna(subset=["mapped_class"])

    n_annotations_all = len(df)
    n_images_all = df["Image Index"].nunique()

    test_images = set(pd.read_csv(TEST_CSV)["Image Index"])
    df = df[df["Image Index"].isin(test_images)].reset_index(drop=True)

    n_annotations_test = len(df)
    n_images_test = df["Image Index"].nunique()

    print(f"[INFO] BBox_List_2017.csv completo: {n_annotations_all} annotazioni su {n_images_all} immagini uniche.")
    print(f"[INFO] Dopo il filtro sul test split: {n_annotations_test} annotazioni "
          f"su {n_images_test} immagini uniche (le uniche usate da questo script).")
    if n_annotations_test == 0:
        raise RuntimeError("Nessuna bounding box cade nel test split -- controlla TEST_CSV.")

    return df[["Image Index", "mapped_class", "x", "y", "w", "h"]]


def transform_bbox_to_model_space(x, y, w, h, orig_w, orig_h):
    """Trasforma il bounding box dallo spazio dell'immagine originale allo
    spazio di input del modello (resize a 420 + center crop a 384),
    coerente con lo shared_tf usato nel resto dell'ablation."""
    sx, sy = CROP_RESIZE / orig_w, CROP_RESIZE / orig_h
    x1, y1 = x * sx, y * sy
    x2, y2 = (x + w) * sx, (y + h) * sy
    crop_offset = (CROP_RESIZE - IMAGE_SIZE) / 2
    x1, y1, x2, y2 = x1 - crop_offset, y1 - crop_offset, x2 - crop_offset, y2 - crop_offset
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(IMAGE_SIZE, x2), min(IMAGE_SIZE, y2)
    return x1, y1, x2, y2


def run_condition(condition_name, seed, bbox_df):
    tag = f"{condition_name}_seed{seed}"
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{tag}.pth")
    out_csv = os.path.join(OUTPUT_DIR, f"bbox_localization_{tag}.csv")
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

    rows = []
    for _, row in tqdm(bbox_df.iterrows(), total=len(bbox_df), desc=f"BBox {tag}"):
        img_path = get_image_path(row["Image Index"])
        if img_path is None:
            continue
        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size
        input_tensor = shared_tf(img_pil).unsqueeze(0).to(device)

        class_idx = CLASSES.index(row["mapped_class"])
        target = [ClassifierOutputTarget(class_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=target)[0, :]

        x1, y1, x2, y2 = transform_bbox_to_model_space(
            row["x"], row["y"], row["w"], row["h"], orig_w, orig_h)
        if x2 <= x1 or y2 <= y1:
            continue  # bbox finita completamente fuori dal crop 384x384

        # pointing-game: il pixel di massima attivazione cade nel box?
        peak_y, peak_x = np.unravel_index(np.argmax(grayscale_cam), grayscale_cam.shape)
        hit = int(x1 <= peak_x <= x2 and y1 <= peak_y <= y2)

        # energy-in-bbox: frazione di massa Grad-CAM dentro il box
        cam_pos = np.clip(grayscale_cam, 0, None)
        total = cam_pos.sum()
        box_mask = np.zeros_like(cam_pos)
        box_mask[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = 1
        energy_in_bbox = float((cam_pos * box_mask).sum() / total) if total > 1e-8 else np.nan

        rows.append({
            "Image Index": row["Image Index"], "condition": condition_name, "seed": seed,
            "class": row["mapped_class"], "pointing_game_hit": hit, "energy_in_bbox": energy_in_bbox,
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[DONE] {out_csv} ({len(rows)} annotazioni valutate, {bbox_df['Image Index'].nunique()} immagini uniche nel test set)")


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boot = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [100 * CI_ALPHA / 2, 100 * (1 - CI_ALPHA / 2)])
    return lo, hi


def summarize(conditions, seeds):
    rows = []
    for cond in conditions:
        for seed in seeds:
            path = os.path.join(OUTPUT_DIR, f"bbox_localization_{cond}_seed{seed}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            if len(df) == 0:
                continue
            pg_lo, pg_hi = bootstrap_ci(df["pointing_game_hit"].values)
            en_lo, en_hi = bootstrap_ci(df["energy_in_bbox"].dropna().values)
            rows.append({
                "condition": cond, "seed": seed, "n_annotations": len(df), "n_unique_images": df["Image Index"].nunique(),
                "pointing_game_accuracy": df["pointing_game_hit"].mean(),
                "pointing_game_CI_95_lo": pg_lo, "pointing_game_CI_95_hi": pg_hi,
                "mean_energy_in_bbox": df["energy_in_bbox"].mean(),
                "energy_in_bbox_CI_95_lo": en_lo, "energy_in_bbox_CI_95_hi": en_hi,
            })
    summary = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "SUMMARY_bbox_localization.csv")
    summary.to_csv(out_path, index=False)
    print(f"\n{'='*100}\nBBOX-BASED LOCALIZATION (ground truth reale, non piu' body-silhouette proxy)\n{'='*100}")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(summary.to_string(index=False))
    print(f"\nFull table: {out_path}")
    print("\n[COME LEGGERE]")
    print("- pointing_game_accuracy: frazione di immagini in cui il picco di attivazione")
    print("  cade dentro il riquadro reale della lesione. Un modello che indovina a caso")
    print("  avrebbe un'accuracy proporzionale all'area del box sull'immagine (tipicamente bassa,")
    print("  spesso <20-30% per box piccoli) -- confronta col chance level, non con 50%.")
    print("- mean_energy_in_bbox: quanta massa Grad-CAM cade nel box reale, analogo alla")
    print("  metrica 'in-body' gia' nel paper ma contro ground truth patologico vero.")
    print("- Confronta C0 vs C6 (e le altre condizioni, se le hai lanciate): questa e' la")
    print("  metrica che risponde direttamente alla critica 'in-body mass non e' localizzazione'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=str, default="C0_baseline,C6_full_combined")
    parser.add_argument("--seeds", type=str, default="42,123,2026")
    args = parser.parse_args()
    conditions = args.conditions.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    print("[INFO] Carico le annotazioni BBox_List_2017.csv...")
    bbox_df = load_bbox_annotations()
    print(f"[INFO] {len(bbox_df)} annotazioni con classe mappabile su una delle 14 patologie del modello.")

    for cond in conditions:
        for seed in seeds:
            run_condition(cond, seed, bbox_df)

    summarize(conditions, seeds)


if __name__ == "__main__":
    main()
