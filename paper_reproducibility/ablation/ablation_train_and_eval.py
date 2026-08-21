"""
ablation_train_and_eval.py

OBIETTIVO
---------
Risolve il problema metodologico piu' grave rimasto nel paper CMIG: nella
versione attuale, il modello "post-intervento" differisce da quello
"pre-intervento" su TRE fattori insieme (crop, loss, schedule), quindi non
si puo' attribuire il cambiamento nella metrica di localizzazione a nessuno
dei tre in particolare -- ne' escludere che sia solo variazione stocastica
del training.

Questo script allena le 7 condizioni richieste dal revisore (baseline,
i tre fattori isolati singolarmente, due combinazioni parziali, la
combinazione completa), ciascuna con piu' seed indipendenti, e valuta
ognuna con lo stesso protocollo. Lo script successivo
(analyze_ablation_results.py) fa l'analisi statistica sui risultati salvati
qui, incluso il confronto di localizzazione Grad-CAM e gli intervalli di
confidenza patient-clustered.

CONDIZIONI (fattori: crop, loss, schedule, augmentation)
----------------------------------------------------------------
  C0_baseline      : crop=NO  loss=ASL  schedule=fixed10+cosine  aug=flip   (= pipeline originale pre-intervento)
  C1_crop_only     : crop=SI  loss=ASL  schedule=fixed10+cosine  aug=flip
  C2_bce_only      : crop=NO  loss=BCE  schedule=fixed10+cosine  aug=flip
  C3_schedule_only : crop=NO  loss=ASL  schedule=earlystop20     aug=flip
  C4_noaug_only    : crop=NO  loss=ASL  schedule=fixed10+cosine  aug=none
  C5_crop_bce      : crop=SI  loss=BCE  schedule=fixed10+cosine  aug=flip
  C6_full_combined : crop=SI  loss=BCE  schedule=earlystop20     aug=none   (= pipeline originale post-intervento)

Tutto il resto (batch size, learning rate, weight decay, ottimizzatore,
backbone/checkpoint) e' tenuto costante fra le condizioni. NOTA: il weight
decay non specificato nello script "post-intervento" originale corrisponde
in realta' al default di AdamW (1e-2), identico a quello esplicitato nello
script "pre-intervento" -- quindi qui e' tenuto fisso a 1e-2 ovunque e NON
e' un fattore dell'ablation (non lo era mai stato realmente).

COSTO COMPUTAZIONALE -- LEGGERE PRIMA DI LANCIARE
----------------------------------------------------------------
7 condizioni x N_SEEDS run, ciascuna fino a 20 epoche di fine-tuning di
ConvNeXt-Base su ~21.500 immagini. Con batch size 16 su una GPU singola
questo e' un carico serio (dell'ordine di giorni, non ore, per l'intero
fattoriale a N_SEEDS=3). Suggerimenti pratici:
  1. Prima lancia con N_SEEDS=1 e MAX_EPOCHS_CAP basso (es. 5) per
     verificare che tutto giri senza errori end-to-end su tutte e 7 le
     condizioni, prima di lanciare la versione completa.
  2. Poi lancia la versione seria con N_SEEDS=3 (minimo statisticamente
     difendibile) o 5 (se il tempo lo consente, come suggerito dalla
     revisione).
  3. Lo script salva un checkpoint per ogni (condizione, seed) e riprende
     da dove interrotto se rilanciato (skip delle combinazioni gia'
     completate) -- puoi quindi lanciarlo a pezzi.

OUTPUT
------
Per ogni (condizione, seed):
  - checkpoint del modello migliore (per macro ROC-AUC su validation)
  - riga di metriche globali su test (native preprocessing) in
    ablation_results/global_metrics.csv
  - predizioni per-immagine su test (per bootstrap successivo) in
    ablation_results/test_predictions_<condition>_seed<seed>.csv
  - metrica di localizzazione Grad-CAM (stesso campione di 1000 immagini,
    stesso preprocessing condiviso 420->384, per confronto equo fra
    condizioni) in ablation_results/localization_<condition>_seed<seed>.csv
"""

import os
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import timm
from pytorch_grad_cam import GradCAM
from scipy import ndimage

# ============================================================
# CONFIGURAZIONE -- adatta questi path alla tua macchina
# ============================================================
ROOT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
TRAIN_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv"
VAL_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
TEST_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
CHECKPOINT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/ablation_checkpoints"
OUTPUT_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/ablation_results"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SEEDS = 3                 # minimo statisticamente difendibile; alza a 5 se hai tempo
SEEDS = [42, 123, 2026][:N_SEEDS]
MAX_EPOCHS_CAP = 20          # tetto assoluto di epoche per qualunque condizione (sicurezza)
LOCALIZATION_N_SAMPLES = 1000
LOCALIZATION_SEED = 42       # stesso campione gia' usato nell'analisi principale del paper

BATCH_SIZE = 16
LR = 3e-5
WEIGHT_DECAY = 1e-2          # costante ovunque, vedi nota sopra
IMAGE_SIZE = 384
CROP_RESIZE = 420            # per le condizioni con crop=SI

CLASSES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
           "Fibrosis", "Pleural_Thickening", "Hernia"]
NUM_CLASSES = len(CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# DEFINIZIONE DELLE 7 CONDIZIONI
# ============================================================
CONDITIONS = {
    "C0_baseline":      dict(crop=False, loss="asl", schedule="fixed10_cosine", aug=True),
    "C1_crop_only":     dict(crop=True,  loss="asl", schedule="fixed10_cosine", aug=True),
    "C2_bce_only":      dict(crop=False, loss="bce", schedule="fixed10_cosine", aug=True),
    "C3_schedule_only": dict(crop=False, loss="asl", schedule="earlystop20",    aug=True),
    "C4_noaug_only":    dict(crop=False, loss="asl", schedule="fixed10_cosine", aug=False),
    "C5_crop_bce":      dict(crop=True,  loss="bce", schedule="fixed10_cosine", aug=True),
    "C6_full_combined": dict(crop=True,  loss="bce", schedule="earlystop20",    aug=False),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        if img_path is None:
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), torch.zeros(NUM_CLASSES), row["Image Index"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label_vec = torch.zeros(NUM_CLASSES)
        for l in str(row["Finding Labels"]).split("|"):
            if l in CLASSES:
                label_vec[CLASSES.index(l)] = 1.0
        return img, label_vec, row["Image Index"]


def build_transforms(cfg):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if cfg["crop"]:
        base = [transforms.Resize((CROP_RESIZE, CROP_RESIZE)), transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE))]
    else:
        base = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]

    train_ops = list(base)
    if cfg["aug"]:
        train_ops.append(transforms.RandomHorizontalFlip(p=0.5))
    train_ops += [transforms.ToTensor(), norm]

    eval_ops = list(base) + [transforms.ToTensor(), norm]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, xs, ys):
        xs_sig = torch.sigmoid(xs)
        xs_pos, xs_neg = xs_sig, 1.0 - xs_sig
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        loss_pos = ys * torch.log(xs_pos.clamp(min=self.eps)) * ((1 - xs_pos) ** self.gamma_pos)
        loss_neg = (1 - ys) * torch.log(xs_neg.clamp(min=self.eps)) * ((1 - xs_neg) ** self.gamma_neg)
        return -1 * (loss_pos + loss_neg).mean()


def build_loss(cfg):
    if cfg["loss"] == "asl":
        return AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    elif cfg["loss"] == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(cfg["loss"])


def evaluate(model, loader):
    model.eval()
    outs, targs, names = [], [], []
    with torch.no_grad():
        for images, targets, img_names in loader:
            o = torch.sigmoid(model(images.to(device))).cpu().numpy()
            outs.append(o)
            targs.append(targets.numpy())
            names.extend(img_names)
    outs = np.vstack(outs)
    targs = np.vstack(targs)
    return outs, targs, names


def compute_global_metrics(outs, targs):
    macro_auc = np.mean([roc_auc_score(targs[:, i], outs[:, i]) for i in range(NUM_CLASSES)])
    micro_auc = roc_auc_score(targs.ravel(), outs.ravel())
    macro_prauc = np.mean([average_precision_score(targs[:, i], outs[:, i]) for i in range(NUM_CLASSES)])
    preds = (outs >= 0.5).astype(int)
    precision = precision_score(targs, preds, average="macro", zero_division=0)
    recall = recall_score(targs, preds, average="macro", zero_division=0)
    f1 = f1_score(targs, preds, average="macro", zero_division=0)
    return dict(macro_auc=macro_auc, micro_auc=micro_auc, macro_prauc=macro_prauc,
                precision=precision, recall=recall, f1=f1)


def train_one(condition_name, cfg, seed):
    tag = f"{condition_name}_seed{seed}"
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{tag}.pth")
    metrics_flag = os.path.join(OUTPUT_DIR, f"DONE_{tag}.json")
    if os.path.exists(metrics_flag):
        print(f"[SKIP] {tag} already completed.")
        return json.load(open(metrics_flag))

    print(f"\n{'='*70}\n[RUN] {tag}  cfg={cfg}\n{'='*70}")
    set_seed(seed)

    train_tf, eval_tf = build_transforms(cfg)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(NIHChestXrayDataset(TRAIN_CSV, train_tf), batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=2, pin_memory=True, generator=g)
    val_loader = DataLoader(NIHChestXrayDataset(VAL_CSV, eval_tf), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(NIHChestXrayDataset(TEST_CSV, eval_tf), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    set_seed(seed)  # re-seed right before model init so weight init is seed-controlled too
    model = timm.create_model('convnext_base.fb_in22k', pretrained=True, num_classes=NUM_CLASSES).to(device)
    criterion = build_loss(cfg)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if cfg["schedule"] == "fixed10_cosine":
        epochs, patience = 10, None
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif cfg["schedule"] == "earlystop20":
        epochs, patience = MAX_EPOCHS_CAP, 8
        scheduler = None
    else:
        raise ValueError(cfg["schedule"])

    best_auc, epochs_no_improve = 0.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for images, targets, _ in tqdm(train_loader, desc=f"{tag} [Ep {epoch}/{epochs}]"):
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = criterion(outputs, targets.to(device))
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        val_outs, val_targs, _ = evaluate(model, val_loader)
        macro_auc = np.mean([roc_auc_score(val_targs[:, i], val_outs[:, i]) for i in range(NUM_CLASSES)])
        print(f"--> {tag} Epoch {epoch}: Val Macro-AUC={macro_auc:.4f}")

        if macro_auc > best_auc:
            best_auc, epochs_no_improve = macro_auc, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if patience is not None and epochs_no_improve >= patience:
                print(f"[STOP] Early stopping at epoch {epoch} for {tag}.")
                break

    # --- valutazione finale sul test set con il checkpoint migliore ---
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_outs, test_targs, test_names = evaluate(model, test_loader)
    metrics = compute_global_metrics(test_outs, test_targs)
    metrics.update(condition=condition_name, seed=seed, best_val_auc=best_auc)

    pred_df = pd.DataFrame(test_outs, columns=[f"prob_{c}" for c in CLASSES])
    pred_df.insert(0, "Image Index", test_names)
    pred_df.to_csv(os.path.join(OUTPUT_DIR, f"test_predictions_{tag}.csv"), index=False)

    json.dump(metrics, open(metrics_flag, "w"), indent=2)
    print(f"[DONE] {tag}: {metrics}")
    return metrics


# ============================================================
# METRICA DI LOCALIZZAZIONE (preprocessing condiviso 420->384 per tutte
# le condizioni, cosi' il confronto e' su pixel identici; vedi discussione
# nel paper sul confound "out-of-distribution" per le condizioni senza crop)
# ============================================================
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


def run_localization(condition_name, seed):
    tag = f"{condition_name}_seed{seed}"
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{tag}.pth")
    out_csv = os.path.join(OUTPUT_DIR, f"localization_{tag}.csv")
    if os.path.exists(out_csv):
        print(f"[SKIP] localization for {tag} already computed.")
        return
    if not os.path.exists(ckpt_path):
        print(f"[WARN] checkpoint missing for {tag}, skipping localization.")
        return

    model = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    cam = GradCAM(model=model, target_layers=[model.stages[-1]])

    shared_tf = transforms.Compose([
        transforms.Resize((CROP_RESIZE, CROP_RESIZE)), transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_only_tf = transforms.Compose([
        transforms.Resize((CROP_RESIZE, CROP_RESIZE)), transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE))
    ])

    df = pd.read_csv(TEST_CSV)
    n = min(LOCALIZATION_N_SAMPLES, len(df))
    sample_df = df.sample(n=n, random_state=LOCALIZATION_SEED).reset_index(drop=True)

    rows = []
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc=f"Localization {tag}"):
        img_path = get_image_path(row["Image Index"])
        if img_path is None:
            continue
        img_pil = Image.open(img_path).convert("RGB")
        input_tensor = shared_tf(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(input_tensor)).cpu().numpy()[0]
        top1 = int(np.argmax(probs))
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(top1)])[0, :]
        mask = body_silhouette_mask(img_only_tf(img_pil))
        cam_pos = np.clip(grayscale_cam, 0, None)
        total = cam_pos.sum()
        frac_inside = float((cam_pos * mask).sum() / total) if total > 1e-8 else np.nan
        rows.append({"Image Index": row["Image Index"], "condition": condition_name, "seed": seed,
                      "top1_class": CLASSES[top1], "fraction_inside": frac_inside})

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[DONE] localization for {tag} -> {out_csv}")


def merge_into_csv(new_rows, csv_path, key_cols):
    """Aggiunge nuove righe a un CSV esistente senza perdere quelle gia' presenti
    (necessario perche' lo script puo' essere lanciato piu' volte, una condizione
    alla volta): se una riga con la stessa chiave esiste gia', viene sostituita."""
    new_df = pd.DataFrame(new_rows)
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        old_df = old_df[~old_df.set_index(key_cols).index.isin(new_df.set_index(key_cols).index)]
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)
    return combined


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Ablation fattoriale (crop/loss/schedule) per il modello ConvNeXt del paper CMIG.")
    parser.add_argument(
        "--conditions", type=str, default="all",
        help="Nomi condizioni separati da virgola (es. 'C0_baseline,C1_crop_only'), oppure 'all' (default).")
    parser.add_argument(
        "--seeds", type=str, default="all",
        help="Seed separati da virgola (es. '42,123'), oppure 'all' per usare tutti quelli in SEEDS (default).")
    parser.add_argument(
        "--phase", type=str, default="all", choices=["train", "localization", "all"],
        help="'train' = solo training+valutazione globale; 'localization' = solo metrica Grad-CAM "
             "(richiede checkpoint gia' presenti); 'all' = entrambe le fasi in sequenza (default).")
    args = parser.parse_args()

    cond_names = list(CONDITIONS.keys()) if args.conditions == "all" else args.conditions.split(",")
    for c in cond_names:
        if c not in CONDITIONS:
            raise ValueError(f"Condizione sconosciuta: {c}. Scegli tra: {list(CONDITIONS.keys())}")
    seeds = SEEDS if args.seeds == "all" else [int(s) for s in args.seeds.split(",")]

    print(f"[CONFIG] Condizioni da eseguire: {cond_names}")
    print(f"[CONFIG] Seed da eseguire: {seeds}")
    print(f"[CONFIG] Fase: {args.phase}")

    if args.phase in ("train", "all"):
        new_metrics = []
        for cond_name in cond_names:
            for seed in seeds:
                new_metrics.append(train_one(cond_name, CONDITIONS[cond_name], seed))
        csv_path = os.path.join(OUTPUT_DIR, "global_metrics.csv")
        merge_into_csv(new_metrics, csv_path, key_cols=["condition", "seed"])
        print(f"\n[SUMMARY] Global metrics updated in {csv_path}")

    if args.phase in ("localization", "all"):
        print("\n[PHASE 2] Computing Grad-CAM localization metric for the selected checkpoints...")
        for cond_name in cond_names:
            for seed in seeds:
                run_localization(cond_name, seed)

    print("\n[DONE] Run analyze_ablation_results.py at any time to see the statistical comparison "
          "for whatever conditions/seeds have completed so far.")


if __name__ == "__main__":
    main()
