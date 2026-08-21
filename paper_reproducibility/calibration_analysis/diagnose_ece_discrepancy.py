"""
diagnose_ece_discrepancy.py

OBIETTIVO
---------
Isola la causa della discrepanza fra:
  - il modello di riferimento originale (single-run, checkpoint della tesi),
    che nel paper riporta ECE macro = 0.1015 (Section 3.6, Table 7);
  - i checkpoint C2/C5/C6 dell'ablation (stessa ricetta: BCE, e per C6 anche
    crop+schedule+no-aug), che riportano ECE macro ~= 0.01, dieci volte piu'
    basso.

Un revisore ha correttamente segnalato che questa discrepanza non puo'
restare "provisional and unreconciled" in un paper Q1: o si risolve la
causa, o si toglie il confronto 29x dal paper. Questo script prova a
risolverla, isolando le cause piu' plausibili UNA ALLA VOLTA:

  1. Il checkpoint originale e' DAVVERO diverso dai checkpoint C6
     dell'ablation? (confronto hash + confronto pesi)
  2. Le predizioni del checkpoint originale, ripassate DA CAPO attraverso
     la stessa identica pipeline di preprocessing e la stessa identica
     funzione di calcolo ECE usata per l'ablation, danno ancora ECE~0.10,
     o improvvisamente danno ECE~0.01? (isola: e' un problema del modello,
     o della VECCHIA pipeline di valutazione/preprocessing?)
  3. Le probabilita' predette hanno una distribuzione sospetta (es. tutte
     vicine a 0.5 = sigmoid applicata due volte; oppure fuori da [0,1] =
     sigmoid mai applicata; oppure quasi tutte 0 o 1 = overconfidence
     estrema, coerente con Asymmetric Loss anche se il nome del file dice
     "debiased"/BCE)?
  4. L'ordine delle classi nel vettore di output coincide fra le due
     pipeline? (uno scambio di colonne produrrebbe un ECE artificialmente
     alto pur con probabilita' individualmente corrette)

CONFIGURAZIONE
---------------
Devi impostare CHECKPOINT_ORIGINAL qui sotto: il path del checkpoint usato
per generare la Table 7 del paper (probabilmente
".../checkpoints/best_debiased_convnext.pth" dal repository originale,
NON dalla cartella ablation_checkpoints/). Se non sei sicuro di quale file
sia, cerca nel tuo codice originale (calibrationConvNeXt-Debiasing.py) la
riga che carica il checkpoint per la calibrazione: il path li' e' quello
giusto.

USO
---
    python diagnose_ece_discrepancy.py

OUTPUT
------
Un report testuale con i risultati dei 4 controlli sopra, e un verdetto
diagnostico finale (quale spiegazione e' la piu' plausibile), piu' un CSV
con le predizioni ricalcolate del checkpoint originale per ispezione
manuale.
"""

import os
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm

# ============================================================
# CONFIGURAZIONE -- ADATTA QUESTO PATH
# ============================================================
CHECKPOINT_ORIGINAL = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_debiased_convnext.pth"
ABLATION_CHECKPOINT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/ablation_checkpoints"
TEST_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
ROOT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
OUTPUT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/ablation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 384
CROP_RESIZE = 420
BATCH_SIZE = 16

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


class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = get_image_path(row["Image Index"])
        img = Image.open(img_path).convert("RGB") if img_path else Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        if self.transform:
            img = self.transform(img)
        label_vec = torch.zeros(NUM_CLASSES)
        for l in str(row["Finding Labels"]).split("|"):
            if l in CLASSES:
                label_vec[CLASSES.index(l)] = 1.0
        return img, label_vec, row["Image Index"]


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(ckpt_path):
    model = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=NUM_CLASSES)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, state


def ece_macro(probs, labels, n_bins=10):
    eces = []
    for c in range(probs.shape[1]):
        p, y = probs[:, c], labels[:, c]
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_idx = np.clip(np.digitize(p, bin_edges) - 1, 0, n_bins - 1)
        ece = 0.0
        for b in range(n_bins):
            mask = bin_idx == b
            n_b = mask.sum()
            if n_b == 0:
                continue
            ece += (n_b / len(p)) * abs(y[mask].mean() - p[mask].mean())
        eces.append(ece)
    return float(np.mean(eces)), eces


def main():
    print("=" * 90)
    print("DIAGNOSTICA DISCREPANZA ECE -- checkpoint originale vs ablation C6")
    print("=" * 90)

    # ---- CHECK 1: confronto file hash e pesi ----
    print("\n[CHECK 1] Confronto checkpoint (hash SHA256 e differenza pesi)")
    if not os.path.exists(CHECKPOINT_ORIGINAL):
        print(f"  [ERRORE] {CHECKPOINT_ORIGINAL} non trovato. Correggi CHECKPOINT_ORIGINAL in cima allo script.")
        return
    hash_orig = sha256_of_file(CHECKPOINT_ORIGINAL)
    print(f"  Originale: {CHECKPOINT_ORIGINAL}\n    sha256={hash_orig[:16]}...")

    ablation_hashes = {}
    for seed in [42, 123, 2026]:
        p = os.path.join(ABLATION_CHECKPOINT_DIR, f"C6_full_combined_seed{seed}.pth")
        if os.path.exists(p):
            ablation_hashes[seed] = sha256_of_file(p)
            same = "IDENTICO all'originale!" if ablation_hashes[seed] == hash_orig else "diverso"
            print(f"  C6 seed {seed}: sha256={ablation_hashes[seed][:16]}...  ({same})")

    model_orig, state_orig = load_model(CHECKPOINT_ORIGINAL)
    if 42 in ablation_hashes:
        model_c6_42, state_c6_42 = load_model(os.path.join(ABLATION_CHECKPOINT_DIR, "C6_full_combined_seed42.pth"))
        # differenza media assoluta dei pesi, layer per layer, riassunta
        total_diff, total_params = 0.0, 0
        for k in state_orig:
            if k in state_c6_42 and state_orig[k].shape == state_c6_42[k].shape:
                diff = (state_orig[k].float() - state_c6_42[k].float()).abs().mean().item()
                total_diff += diff * state_orig[k].numel()
                total_params += state_orig[k].numel()
        print(f"  Differenza media assoluta dei pesi (originale vs C6 seed42): {total_diff/max(total_params,1):.6f}")
        print("  (valori vicini a 0 indicano pesi quasi identici; valori tipici fra modelli")
        print("   allenati indipendentemente con lo stesso recipe sono nell'ordine di 0.01-0.1)")

    # ---- CHECK 2: ricalcolo ECE del checkpoint originale con la pipeline nuova ----
    print("\n[CHECK 2] Ricalcolo ECE del checkpoint ORIGINALE con la pipeline di valutazione dell'ablation")
    print("  (stesso preprocessing 420->384, stesso test_split.csv, stessa funzione ECE)")
    tf = transforms.Compose([
        transforms.Resize((CROP_RESIZE, CROP_RESIZE)), transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    loader = DataLoader(NIHChestXrayDataset(TEST_CSV, tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs, all_labels, all_names = [], [], []
    with torch.no_grad():
        for images, labels, names in tqdm(loader, desc="Inferenza checkpoint originale"):
            probs = torch.sigmoid(model_orig(images.to(device))).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            all_names.extend(names)
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    macro_ece, per_class_ece = ece_macro(all_probs, all_labels)
    print(f"\n  ECE macro ricalcolato per il checkpoint ORIGINALE, pipeline nuova: {macro_ece:.4f}")
    print(f"  (per confronto: 0.1015 e' quello riportato nel paper con la vecchia pipeline;")
    print(f"   ~0.01 e' quello dei checkpoint C6 dell'ablation)")

    pred_df = pd.DataFrame(all_probs, columns=[f"prob_{c}" for c in CLASSES])
    pred_df.insert(0, "Image Index", all_names)
    out_csv = os.path.join(OUTPUT_DIR, "diagnostic_original_checkpoint_predictions.csv")
    pred_df.to_csv(out_csv, index=False)

    # ---- CHECK 3: statistiche sospette sulle probabilita' ----
    print("\n[CHECK 3] Statistiche delle probabilita' predette (originale, pipeline nuova)")
    print(f"  Min={all_probs.min():.4f}  Max={all_probs.max():.4f}  Media={all_probs.mean():.4f}")
    print(f"  Frazione di valori in [0.45, 0.55] (sospetto sigmoid doppia): {((all_probs>=0.45)&(all_probs<=0.55)).mean():.3f}")
    print(f"  Frazione di valori < 0.001 o > 0.999 (overconfidence estrema): "
          f"{((all_probs<0.001)|(all_probs>0.999)).mean():.3f}")
    if all_probs.min() < 0 or all_probs.max() > 1:
        print("  [ALLARME] Valori fuori da [0,1] -- la sigmoid potrebbe non essere applicata da qualche parte.")

    # ---- CHECK 4: confronto diretto con le predizioni gia' salvate per C6 (se disponibili) ----
    print("\n[CHECK 4] Confronto diretto con le predizioni gia' salvate per C6 seed 42 (se presenti)")
    c6_pred_path = os.path.join(OUTPUT_DIR, "test_predictions_C6_full_combined_seed42.csv")
    if os.path.exists(c6_pred_path):
        c6_preds = pd.read_csv(c6_pred_path).set_index("Image Index")
        merged = pred_df.set_index("Image Index").join(c6_preds, lsuffix="_orig", rsuffix="_c6", how="inner")
        for c in CLASSES[:3]:
            corr = merged[f"prob_{c}_orig"].corr(merged[f"prob_{c}_c6"])
            print(f"  Correlazione probabilita' predette, classe {c}: {corr:.3f} "
                  f"(vicino a 1 = i due modelli si comportano in modo molto simile)")
    else:
        print(f"  [SKIP] {c6_pred_path} non trovato -- lancia prima full_calibration_reanalysis.py per generarlo.")

    # ---- VERDETTO ----
    print("\n" + "=" * 90)
    print("VERDETTO DIAGNOSTICO")
    print("=" * 90)
    if abs(macro_ece - 0.01) < 0.03:
        print("-> Il checkpoint ORIGINALE, ripassato dalla pipeline nuova, da' un ECE vicino a 0.01,")
        print("   NON a 0.1015. Questo indica che il problema NON e' nel modello (i pesi calibrano bene),")
        print("   ma nella VECCHIA pipeline di valutazione usata per calcolare 0.1015 (probabilmente un")
        print("   preprocessing diverso, o un bug nello script di calibrazione originale).")
        print("   AZIONE CONSIGLIATA: ricalcola la Table 7 del paper con la pipeline nuova (questo script")
        print("   ha gia' prodotto le predizioni corrette in diagnostic_original_checkpoint_predictions.csv)")
        print("   e aggiorna il valore 0.1015 -> ~0.01 in tutto il paper. Il confronto 29-fold resta valido")
        print("   E ora e' pienamente riconciliato.")
    elif abs(macro_ece - 0.1015) < 0.03:
        print("-> Il checkpoint ORIGINALE, ripassato dalla pipeline nuova, conferma ECE~0.10, coerente")
        print("   col valore gia' nel paper. Il problema NON e' nella pipeline di valutazione:")
        print("   il checkpoint originale calibra genuinamente peggio dei checkpoint C6 dell'ablation,")
        print("   nonostante nominalmente la stessa ricetta (crop+BCE+schedule+no-aug).")
        print("   Controlla il CHECK 1 sopra: se la differenza media dei pesi e' grande, il checkpoint")
        print("   originale potrebbe essere stato allenato con una configurazione leggermente diversa da")
        print("   quella che il codice attuale implementa (es. una versione precedente dello script).")
        print("   AZIONE CONSIGLIATA: se non riesci a spiegare la differenza, rimuovi il confronto")
        print("   29-fold dal ruolo di 'largest effect' nel paper e mantieni solo la calibrazione")
        print("   descrittiva del modello di riferimento (opzione B suggerita dal revisore).")
    else:
        print(f"-> Il valore ricalcolato ({macro_ece:.4f}) non e' vicino a nessuno dei due regimi noti.")
        print("   Serve un'ispezione manuale delle predizioni salvate in")
        print(f"   {out_csv}")
        print("   per capire cosa sta succedendo.")


if __name__ == "__main__":
    main()
