import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, precision_recall_fscore_support

# =====================================================
# CONFIGURAZIONE PATHS
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"

model_path = "best_densenet121_v3_512px_asl.pth"
thresholds_path = "optimized_thresholds_v3.json"

# =====================================================
# COSTANTI
# =====================================================
IMAGE_SIZE = 512

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)

# =====================================================
# UTILS
# =====================================================
def encode_labels(label_str):
    vec = torch.zeros(num_classes)
    labels = label_str.split("|")
    for l in labels:
        if l in classes:
            vec[classes.index(l)] = 1.0
    return vec

def get_image_path(img_name):
    for i in range(1, 13):
        folder = f"images_{i:03d}"
        path = os.path.join(root_dir, folder, "images", img_name)
        if os.path.exists(path):
            return path
    return None

class NIHChestDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        label_str = row["Finding Labels"]
        img_path = get_image_path(img_name)
        if img_path is None: return None, None
        image = Image.open(img_path).convert("RGB")
        label = encode_labels(label_str)
        if self.transform: image = self.transform(image)
        return image, label

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0: return torch.empty(0), torch.empty(0)
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return images, labels

# =====================================================
# LOAD DATA & THRESHOLDS
# =====================================================
with open(thresholds_path, "r") as f:
    optimized_thresholds = json.load(f)

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_df = pd.read_csv(test_csv)
test_loader = DataLoader(
    NIHChestDataset(test_df, test_transform),
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn
)

# =====================================================
# LOAD MODEL
# =====================================================
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
print(f"\nModello DenseNet121 v3 caricato per il Test definitivo ✔")

# =====================================================
# INFERENCE ON TEST SET
# =====================================================
all_probs = []
all_targets = []

print("Esecuzione inferenza sul TEST SET...")
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        if images.numel() == 0: continue
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

all_probs = np.vstack(all_probs)
all_targets = np.vstack(all_targets)

# =====================================================
# SANITIZZAZIONE TIPI DATI
# =====================================================
all_targets = all_targets.astype(np.int32)
all_probs = all_probs.astype(np.float32)

preds_bin = np.zeros_like(all_probs)
for j, c in enumerate(classes):
    t = optimized_thresholds[c]
    preds_bin[:, j] = (all_probs[:, j] >= t).astype(int)

# =====================================================
# CALCOLO METRICHE GLOBALI E PER CLASSE
# =====================================================
print("\n" + "="*50)
print("FINAL METRICS SUMMARY (TEST SET - DenseNet121 v3 ImageNet 512px)")
print("="*50)

# 1. ROC-AUC (Macro e Micro)
auc_dict = {}
auc_list = []
for j, c in enumerate(classes):
    try:
        auc_c = roc_auc_score(all_targets[:, j], all_probs[:, j])
        auc_list.append(auc_c)
        auc_dict[c] = float(auc_c)
    except ValueError:
        auc_list.append(0.0)
        auc_dict[c] = 0.0

macro_auc = np.mean(auc_list)
micro_auc = roc_auc_score(all_targets, all_probs, average="micro")

# 2. PR-AUC Macro e metriche puntuali
pr_auc_list = []
precision_cls, recall_cls, f1_cls, _ = precision_recall_fscore_support(
    all_targets, preds_bin, average=None, zero_division=0
)

for j in range(num_classes):
    precision_vals, recall_vals, _ = precision_recall_curve(all_targets[:, j], all_probs[:, j])
    pr_auc_list.append(auc(recall_vals, precision_vals))
macro_prauc = np.mean(pr_auc_list)

macro_precision = np.mean(precision_cls)
macro_recall = np.mean(recall_cls)
macro_f1_val = np.mean(f1_cls)

# Report di classificazione standard
report_dict = classification_report(all_targets, preds_bin, target_names=classes, output_dict=True, zero_division=0)

# =====================================================
# STAMPA TABELLA DETTAGLIO PER PATOLOGIA
# =====================================================
print("\n" + "="*95)
print(f"{'Patologia':20s} | {'Soglia':8s} | {'ROC-AUC':8s} | {'PR-AUC':8s} | {'Precision':9s} | {'Recall':8s} | {'F1-Score':8s}")
print("="*95)

for j, c in enumerate(classes):
    thr = optimized_thresholds[c]
    roc = auc_list[j]
    pr = pr_auc_list[j]
    prec = precision_cls[j]
    rec = recall_cls[j]
    f1 = f1_cls[j]
    
    print(f"{c:20s} | {thr:8.4f} | {roc:8.4f} | {pr:8.4f} | {prec:9.4f} | {rec:8.4f} | {f1:8.4f}")

print("="*95)

# =====================================================
# STAMPA METRICHE GLOBALI (FORMATO COERENTE)
# =====================================================
print("\n" + "="*45)
print(f"{'Metrica Globale':25s} | {'Valore':8s}")
print("="*45)
print(f"{'Media Macro ROC-AUC':25s} | {macro_auc:8.4f}")
print(f"{'Media Micro ROC-AUC':25s} | {micro_auc:8.4f}")
print(f"{'Macro PR-AUC':25s} | {macro_prauc:8.4f}")
print(f"{'Macro Precision':25s} | {macro_precision:8.4f}")
print(f"{'Macro Recall':25s} | {macro_recall:8.4f}")
print(f"{'Macro F1-Score':25s} | {macro_f1_val:8.4f}")
print("="*45)

# =====================================================
# SALVATAGGIO RISULTATI STRUTTURATI
# =====================================================
output_results = {
    "macro_auc": float(macro_auc),
    "micro_auc": float(micro_auc),
    "macro_prauc": float(macro_prauc),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1_val),
    "auc_per_class": auc_dict,
    "classification_report": report_dict
}

os.makedirs("checkpoints", exist_ok=True)
with open("checkpoints/final_test_metrics_v3.json", "w") as f:
    json.dump(output_results, f, indent=4)
print("\nTutte le metriche globali e disaggregate sono state salvate in checkpoints/final_test_metrics_v3.json ✔")