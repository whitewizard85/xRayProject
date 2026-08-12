import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, classification_report, roc_curve

# CONFIGURAZIONE PATHS
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
path_pesi_b7 = "/home/gpuvm/Desktop/Luca Migliaccio/best_efficientnet_b7_asl.pth"

IMAGE_SIZE = 600 
BATCH_SIZE = 16

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)

def encode_labels(label_str):
    vec = torch.zeros(num_classes)
    labels = str(label_str).split("|")
    for l in labels:
        if l in classes: vec[classes.index(l)] = 1.0
    return vec

def get_image_path(img_name):
    for i in range(1, 13):
        folder = f"images_{i:03d}"
        path = os.path.join(root_dir, folder, "images", img_name)
        if os.path.exists(path): return path
    return None

class NIHChestDatasetB7(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = get_image_path(img_name)
        if img_path is None: return None, None, None
        try:
            image = Image.open(img_path).convert("RGB")
            label = encode_labels(row["Finding Labels"])
            if self.transform: image = self.transform(image)
            return image, label, img_name
        except Exception: return None, None, None

def collate_fn(batch):
    batch = [b for b in batch if b is not None and b[0] is not None]
    if len(batch) == 0: return torch.empty(0), torch.empty(0), []
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]), [b[2] for b in batch]

# INIZIALIZZAZIONE MODELLO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Caricamento EfficientNet-B7...")
model = models.efficientnet_b7(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(path_pesi_b7, map_location=device))
model = model.to(device)
model.eval()

val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_loader = DataLoader(NIHChestDatasetB7(val_df, test_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)
test_loader = DataLoader(NIHChestDatasetB7(test_df, test_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

# 1. VALIDAZIONE PER CALCOLO SOGLIE
print("\nEstrazione probabilità sul Validation Set per calcolo soglie... 🔍")
val_preds, val_targets = [], []
with torch.no_grad():
    for images, labels, _ in tqdm(val_loader, desc="Validation"):
        if images.numel() == 0: continue
        images = images.to(device)
        with torch.amp.autocast('cuda'):
            outputs = torch.sigmoid(model(images))
        val_preds.append(outputs.cpu().numpy())
        val_targets.append(labels.numpy())

val_preds = np.vstack(val_preds)
val_targets = np.vstack(val_targets)

best_thresholds = []
print("\n--- SOGLIE OTTIME CALCOLATE (INDICE DI YOUDEN) ---")
for j, c in enumerate(classes):
    fpr, tpr, thresholds = roc_curve(val_targets[:, j], val_preds[:, j])
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_thresholds.append(thresholds[best_idx])
    print(f"-> {c:<20} : Soglia Ottima = {thresholds[best_idx]:.4f}")

# 2. INFERENZA COMPLETA SUL TEST SET
print("\nEsecuzione inferenza sul Test Set...")
test_preds, test_targets, test_names = [], [] , []
with torch.no_grad():
    for images, labels, names in tqdm(test_loader, desc="Test Inference"):
        if images.numel() == 0: continue
        images = images.to(device)
        with torch.amp.autocast('cuda'):
            outputs = torch.sigmoid(model(images))
        test_preds.append(outputs.cpu().numpy())
        test_targets.append(labels.numpy())
        test_names.extend(names)

test_preds = np.vstack(test_preds)
test_targets = np.vstack(test_targets)

test_preds_bin = np.zeros_like(test_preds)
for j in range(num_classes):
    test_preds_bin[:, j] = (test_preds[:, j] >= best_thresholds[j]).astype(int)

# 3. CALCOLO METRICHE REALI PER CLASSE
test_aucs = []
for j in range(num_classes):
    try: test_aucs.append(roc_auc_score(test_targets[:, j], test_preds[:, j]))
    except ValueError: test_aucs.append(0.5)

report_dict = classification_report(test_targets.astype(int), test_preds_bin.astype(int), target_names=classes, zero_division=0, output_dict=True)

# STAMPA DELLA TABELLA EXCEL REALE
print("\n" + "="*90)
print("➔ TABELLA REALE GENERATA PER EXCEL (COPIA E INCOLLA DIRETTAMENTE SU A1)")
print("="*90)
print("Classe\tSoglia Ottima\tROC-AUC REALE\tPrecision\tRecall\tF1-Score\tSupporto")
for j, c in enumerate(classes):
    print(f"{c}\t{best_thresholds[j]:.4f}\t{test_aucs[j]:.4f}\t{report_dict[c]['precision']:.2f}\t{report_dict[c]['recall']:.2f}\t{report_dict[c]['f1-score']:.2f}\t{report_dict[c]['support']}")
print("---" + "\t---"*6)
print(f"Macro Average\t-\t{np.mean(test_aucs):.4f}\t{report_dict['macro avg']['precision']:.2f}\t{report_dict['macro avg']['recall']:.2f}\t{report_dict['macro avg']['f1-score']:.2f}\t{report_dict['macro avg']['support']}")
print("="*90)

# 4. ERROR ANALYSIS AUTOMATICA
error_records = []
for idx, img_name in enumerate(test_names):
    for j, c in enumerate(classes):
        prob = float(test_preds[idx, j])
        target = int(test_targets[idx, j])
        thresh = best_thresholds[j]
        pred = 1 if prob >= thresh else 0
        
        if pred != target:
            err_type = "FN" if target == 1 else "FP"
            err_margin = float(thresh - prob) if err_type == "FN" else float(prob - thresh)
            error_records.append({
                "Image_Index": img_name, "Class": c, "Type": err_type,
                "Probability": prob, "Threshold": thresh, "Target": target, "Error_Margin": err_margin
            })

df_errors = pd.DataFrame(error_records)
df_errors.to_csv("checkpoints/error_analysis_efficientnet_b7.csv", index=False)

df_fn = df_errors[df_errors["Type"] == "FN"].sort_values(by="Error_Margin", ascending=False).head(5)
df_fp = df_errors[df_errors["Type"] == "FP"].sort_values(by="Error_Margin", ascending=False).head(5)

print("\n--- TOP 5 PEGGIORI FALSI NEGATIVI (Mancati) ---")
print(df_fn.to_string(index=False) if not df_fn.empty else "Nessuno")
print("\n--- TOP 5 PEGGIORI FALSI POSITIVI (Falsi Allarmi) ---")
print(df_fp.to_string(index=False) if not df_fp.empty else "Nessuno")