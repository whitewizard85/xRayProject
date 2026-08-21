"""
quantitative_localization_debiasing.py

OBIETTIVO
---------
Colma l'ultimo gap quantitativo del paper CMIG: misura, su un campione
consistente e riproducibile, quanta parte dell'attivazione Grad-CAM cade
DENTRO la sagoma del corpo del paziente (proxy della gabbia toracica)
rispetto a quanta cade FUORI (sfondo, marker, bordi) -- confrontando il
modello ConvNeXt PRIMA e DOPO il debiasing, sulle STESSE 1000 immagini
di test gia' usate in grad-camConvNeXt-DebiasingFinale.py (stesso seed=42,
stesso file di split), cosi' il confronto e' direttamente comparabile con
i report visivi gia' generati per la tesi/paper.

METRICA
-------
Per ciascuna immagine e ciascun modello:
    fraction_inside = sum(cam * mask_corpo) / sum(cam)
dove `cam` e' la heatmap Grad-CAM normalizzata in [0,1] e `mask_corpo` e'
una maschera binaria della silhouette del paziente (vedi NOTA METODOLOGICA).

Il target-class per il Grad-CAM e' la classe Top-1 predetta dal modello
DEBIASED (stessa immagine, stessa classe usata per ENTRAMBI i modelli),
cosi' il confronto pre/post e' sulla stessa domanda ("perche' il modello
pensa X") e non confuso da un cambio di classe spiegata.

NOTA METODOLOGICA (da riportare nel paper / discutere con il relatore)
------------------------------------------------------------------
Non esiste nel repository un modello di segmentazione polmonare dedicato,
quindi la maschera usata qui è una SAGOMA CORPOREA (body silhouette),
ottenuta con soglia di Otsu + componente connessa piu' grande + erosione,
NON una segmentazione polmonare anatomica precisa. E' un proxy
ragionevole per l'obiettivo (i marker/bordi/lettere dello shortcut
learning stanno tipicamente fuori dal corpo o esattamente sul bordo, che
l'erosione esclude), ma va dichiarato esplicitamente come limitazione nel
paper: una vera maschera polmonare (es. rete di segmentazione pre-addestrata
tipo torchxrayvision.baseline_models.chestx_det o simili) darebbe una
misura anatomicamente piu' precisa e andrebbe citata come sviluppo futuro
se i tempi lo consentono.

OUTPUT
------
1. CSV con una riga per immagine: filename, top1_class, fraction_inside_pre,
   fraction_inside_post, delta.
2. Riepilogo statistico stampato a schermo (media, std, test di Wilcoxon
   appaiato) -- pronto da incollare nella Sezione 3.4/Results del paper.
3. 6 immagini di controllo qualitativo (originale + maschera + CAM pre/post)
   per verificare visivamente che la maschera sia sensata prima di fidarsi
   del numero aggregato.

USO
---
Adatta le variabili in CONFIGURAZIONE qui sotto ai path della tua macchina
(sono gli stessi path gia' usati negli script esistenti del repo), poi:

    python quantitative_localization_debiasing.py

Richiede la stessa GPU/ambiente gia' usato per il training (checkpoints
.pth di ConvNeXt pre e post debiasing devono esistere sul disco).
"""

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from scipy import ndimage
from scipy.stats import wilcoxon
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURAZIONE -- adatta questi path alla tua macchina
# ============================================================
ROOT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
TEST_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"

CHECKPOINT_PRE = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_convnext_base_22k.pth"
CHECKPOINT_POST = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_debiased_convnext.pth"

OUTPUT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/Analisi_Localizzazione_Quantitativa"
N_SAMPLES = 1000          # stesso N del report visivo gia' generato
RANDOM_SEED = 42          # stesso seed gia' usato -> stesse identiche immagini
N_QC_EXAMPLES = 6         # quante immagini di controllo visivo salvare

IMAGE_SIZE = 384
BATCH_LOAD_SIZE = 420     # resize prima del center-crop (come evalConvNeXt-Debiasing.py)

CLASSES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
           "Fibrosis", "Pleural_Thickening", "Hernia"]
NUM_CLASSES = len(CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((BATCH_LOAD_SIZE, BATCH_LOAD_SIZE)),
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_image_path(img_name: str):
    for i in range(1, 13):
        path = os.path.join(ROOT_DIR, f"images_{i:03d}", "images", img_name)
        if os.path.exists(path):
            return path
    return None


def load_model(checkpoint_path: str):
    model = timm.create_model("convnext_base.fb_in22k", pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model


class ClassifierOutputTarget:
    """Stesso target-wrapper usato negli script Grad-CAM del progetto."""
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if model_output.ndim == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def body_silhouette_mask(pil_img_gray: Image.Image, erosion_px: int = 6) -> np.ndarray:
    """
    Stima la sagoma corporea del paziente sulla radiografia (proxy della
    gabbia toracica) via soglia di Otsu + componente connessa piu' grande
    + riempimento buchi + erosione del bordo.

    Ritorna una maschera binaria (IMAGE_SIZE x IMAGE_SIZE), 1 = dentro il
    corpo, 0 = sfondo/bordo/marker.
    """
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_erosion, disk

    arr = np.array(pil_img_gray.convert("L"), dtype=np.float64)
    thresh = threshold_otsu(arr)
    mask = arr > thresh

    # tiene solo la componente connessa piu' grande (il corpo del paziente,
    # non frammenti isolati di marker/rumore)
    labeled, n = ndimage.label(mask)
    if n > 0:
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        largest_label = np.argmax(sizes) + 1
        mask = labeled == largest_label

    mask = ndimage.binary_fill_holes(mask)
    if erosion_px > 0:
        mask = binary_erosion(mask, footprint=disk(erosion_px))

    return mask.astype(np.float32)


def fraction_inside(cam: np.ndarray, mask: np.ndarray) -> float:
    cam = np.clip(cam, 0, None)
    total = cam.sum()
    if total <= 1e-8:
        return np.nan
    return float((cam * mask).sum() / total)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(TEST_CSV)
    n = min(N_SAMPLES, len(df))
    sample_df = df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"[INFO] Campione: {n} immagini (seed={RANDOM_SEED}), identico a quello "
          f"gia' usato per i report visivi a 1000 campioni.")

    print("[INFO] Carico i due checkpoint ConvNeXt (pre/post debiasing)...")
    model_pre = load_model(CHECKPOINT_PRE)
    model_post = load_model(CHECKPOINT_POST)

    # target layer: uso lo stesso layer per entrambi i modelli (l'ultimo
    # blocco convoluzionale) per un confronto omogeneo, coerente con
    # grad-camConvNeXt-DebiasingFinale.py
    cam_pre = GradCAM(model=model_pre, target_layers=[model_pre.stages[-1]])
    cam_post = GradCAM(model=model_post, target_layers=[model_post.stages[-1]])

    results = []
    qc_saved = 0

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Analisi"):
        img_name = row["Image Index"]
        img_path = get_image_path(img_name)
        if img_path is None:
            continue

        try:
            img_pil = Image.open(img_path).convert("RGB")
            input_tensor = transform(img_pil).unsqueeze(0).to(device)

            # Top-1 del modello DEBIASED: e' la classe che spieghiamo per
            # ENTRAMBI i modelli, cosi' il confronto e' sulla stessa domanda
            with torch.no_grad():
                out_post = torch.sigmoid(model_post(input_tensor)).cpu().numpy()[0]
            top1_idx = int(np.argmax(out_post))
            top1_conf_post = float(out_post[top1_idx])

            target = [ClassifierOutputTarget(top1_idx)]
            grayscale_cam_pre = cam_pre(input_tensor=input_tensor, targets=target)[0, :]
            grayscale_cam_post = cam_post(input_tensor=input_tensor, targets=target)[0, :]

            # maschera corporea calcolata sull'immagine effettivamente in
            # input al modello (stesso resize/crop), cosi' e' allineata
            # pixel-per-pixel con la CAM
            img_for_mask = transforms.Compose([
                transforms.Resize((BATCH_LOAD_SIZE, BATCH_LOAD_SIZE)),
                transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
            ])(img_pil)
            mask = body_silhouette_mask(img_for_mask)

            frac_pre = fraction_inside(grayscale_cam_pre, mask)
            frac_post = fraction_inside(grayscale_cam_post, mask)

            results.append({
                "Filename": img_name,
                "Top1_Class_Debiased": CLASSES[top1_idx],
                "Top1_Confidence_Debiased": top1_conf_post,
                "Fraction_Inside_PreDebiasing": frac_pre,
                "Fraction_Inside_PostDebiasing": frac_post,
                "Delta": frac_post - frac_pre if (frac_pre == frac_pre and frac_post == frac_post) else np.nan,
            })

            # salva qualche esempio di controllo visivo, per verificare
            # a occhio che la maschera "body silhouette" sia sensata
            if qc_saved < N_QC_EXAMPLES:
                fig, ax = plt.subplots(1, 4, figsize=(16, 4))
                img_np = np.array(img_for_mask).astype(float) / 255.0
                ax[0].imshow(img_np); ax[0].set_title("Original"); ax[0].axis("off")
                ax[1].imshow(mask, cmap="gray"); ax[1].set_title("Body mask"); ax[1].axis("off")
                ax[2].imshow(img_np); ax[2].imshow(grayscale_cam_pre, cmap="jet", alpha=0.5)
                ax[2].set_title(f"Pre-debias ({frac_pre:.2f} inside)"); ax[2].axis("off")
                ax[3].imshow(img_np); ax[3].imshow(grayscale_cam_post, cmap="jet", alpha=0.5)
                ax[3].set_title(f"Post-debias ({frac_post:.2f} inside)"); ax[3].axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"qc_example_{qc_saved+1}_{img_name}.png"),
                             dpi=200, bbox_inches="tight")
                plt.close()
                qc_saved += 1

        except Exception as e:
            print(f"[WARN] Errore su {img_name}: {e}")
            continue

    # ------------------------------------------------------------------
    # Aggregazione e statistica
    # ------------------------------------------------------------------
    res_df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "localizzazione_quantitativa_pre_post.csv")
    res_df.to_csv(csv_path, index=False)

    valid = res_df.dropna(subset=["Fraction_Inside_PreDebiasing", "Fraction_Inside_PostDebiasing"])
    mean_pre = valid["Fraction_Inside_PreDebiasing"].mean()
    std_pre = valid["Fraction_Inside_PreDebiasing"].std()
    mean_post = valid["Fraction_Inside_PostDebiasing"].mean()
    std_post = valid["Fraction_Inside_PostDebiasing"].std()

    # test di Wilcoxon appaiato (non parametrico, robusto a distribuzioni
    # non gaussiane della frazione, che e' bounded in [0,1])
    stat, p_value = wilcoxon(valid["Fraction_Inside_PreDebiasing"],
                              valid["Fraction_Inside_PostDebiasing"])

    print("\n" + "=" * 70)
    print("RISULTATO -- frazione media di massa Grad-CAM dentro la sagoma corporea")
    print("=" * 70)
    print(f"N immagini valide analizzate: {len(valid)} / {n}")
    print(f"Pre-debiasing:  media = {mean_pre:.4f}  (std = {std_pre:.4f})")
    print(f"Post-debiasing: media = {mean_post:.4f}  (std = {std_post:.4f})")
    print(f"Delta medio:    {mean_post - mean_pre:+.4f}")
    print(f"Wilcoxon signed-rank test (paired): statistic={stat:.2f}, p-value={p_value:.3e}")
    print("=" * 70)
    print(f"\nCSV completo salvato in: {csv_path}")
    print(f"Esempi di controllo visivo salvati in: {OUTPUT_DIR}")
    print("\n[NEXT STEP] Incolla media/std/p-value qui sopra nella Sezione 3.4 "
          "(Explainability-driven debiasing) e nell'Abstract del paper, sostituendo "
          "l'OPEN POINT relativo alla metrica di localizzazione aggregata.")


if __name__ == "__main__":
    main()

