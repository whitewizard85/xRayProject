import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_recall_fscore_support,
    f1_score
)

from dataset import NIHChestXrayDataset, collate_fn

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n========================")
print("DEVICE")
print("========================")
print(device)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode"

TEST_CSV = os.path.join(BASE_DIR, "test_split.csv")

IMAGE_ROOT = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "Advanced_ResNet50_v1.pth"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "optimized_thresholds_ResNet50_v1.json"
)

# =====================================================
# CLASSES
# =====================================================

classes = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax",
    "Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia","No Finding"
]

valid_classes = classes[:14]
num_classes = len(classes)

# =====================================================
# TRANSFORM
# =====================================================

IMAGE_SIZE = 320

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =====================================================
# DATA
# =====================================================

test_df = pd.read_csv(TEST_CSV)

test_dataset = NIHChestXrayDataset(
    test_df,
    image_root=IMAGE_ROOT,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    collate_fn=collate_fn
)

print("\nTest samples:", len(test_dataset))
print("Test batches:", len(test_loader))

# =====================================================
# LOAD MODEL
# =====================================================

print("\n========================")
print("LOADING MODEL")
print("========================")

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, num_classes)
)

model.load_state_dict(checkpoint["model"])
model = model.to(device)
model.eval()

print("Model loaded ✔")

# =====================================================
# LOAD THRESHOLDS
# =====================================================

print("Thresholds loaded ✔")

with open(THRESHOLD_PATH, "r") as f:
    thresholds = json.load(f)

# =====================================================
# INFERENCE
# =====================================================

print("\n========================")
print("RUNNING INFERENCE")
print("========================")

all_preds = []
all_targets = []

with torch.no_grad():

    for images, labels in tqdm(test_loader):

        if images is None:
            continue

        images = images.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)

        all_preds.append(probs.cpu().numpy())
        all_targets.append(labels.numpy())

preds = np.concatenate(all_preds)
targets = np.concatenate(all_targets)

print("\nPredictions:", preds.shape)
print("Targets:", targets.shape)

# =====================================================
# ROC-AUC PER CLASS
# =====================================================

print("\n========================")
print("ROC-AUC PER CLASS")
print("========================")

roc_scores = []

for i, c in enumerate(valid_classes):

    try:
        roc = roc_auc_score(targets[:, i], preds[:, i])
    except:
        roc = 0.0

    roc_scores.append(roc)

    print(f"{c:22s} {roc:.4f}")

print("\n========================")
print("GLOBAL RESULTS")
print("========================")

mean_roc = np.mean(roc_scores)

pr_auc = average_precision_score(targets[:, :14], preds[:, :14], average="macro")

# =====================================================
# APPLY THRESHOLDS
# =====================================================

binary_preds = np.zeros_like(preds[:, :14])

for i, c in enumerate(valid_classes):

    thr = thresholds[c]
    binary_preds[:, i] = (preds[:, i] >= thr).astype(int)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

report = classification_report(
    targets[:, :14],
    binary_preds,
    target_names=valid_classes,
    zero_division=0
)

print(report)

# =====================================================
# MICRO / MACRO / SAMPLES F1
# =====================================================

micro_f1 = f1_score(targets[:, :14], binary_preds, average="micro")
macro_f1 = f1_score(targets[:, :14], binary_preds, average="macro")
samples_f1 = f1_score(targets[:, :14], binary_preds, average="samples")

print("\n========================")
print("FINAL METRICS")
print("========================")

print("Mean ROC-AUC :", round(mean_roc, 4))
print("PR-AUC       :", round(pr_auc, 4))
print("Macro F1     :", round(macro_f1, 4))
print("Micro F1     :", round(micro_f1, 4))
print("Samples F1   :", round(samples_f1, 4))

# =====================================================
# STAMPA TABELLA DETTAGLIO PER PATOLOGIA
# =====================================================

precision_cls, recall_cls, f1_cls, _ = precision_recall_fscore_support(
    targets[:, :14], binary_preds, average=None, zero_division=0
)

print("\n" + "="*95)
print(f"{'Patologia':20s} | {'Soglia':8s} | {'ROC-AUC':8s} | {'PR-AUC':8s} | {'Precision':9s} | {'Recall':8s} | {'F1-Score':8s}")
print("="*95)

pr_scores_list = []
for i, c in enumerate(valid_classes):
    thr = thresholds[c]
    roc = roc_scores[i]
    
    try:
        pr = average_precision_score(targets[:, i], preds[:, i])
    except:
        pr = 0.0
    pr_scores_list.append(pr)
        
    prec = precision_cls[i]
    rec = recall_cls[i]
    f1 = f1_cls[i]
    
    print(f"{c:20s} | {thr:8.4f} | {roc:8.4f} | {pr:8.4f} | {prec:9.4f} | {rec:8.4f} | {f1:8.4f}")

print("="*95)

# =====================================================
# STAMPA METRICHE GLOBALI (COME DA IMMAGINE)
# =====================================================

# Calcoli globali richiesti
macro_roc_auc = np.mean(roc_scores)
try:
    micro_roc_auc = roc_auc_score(targets[:, :14], preds[:, :14], average="micro")
except:
    micro_roc_auc = 0.0

macro_pr_auc = np.mean(pr_scores_list)
macro_precision = np.mean(precision_cls)
macro_recall = np.mean(recall_cls)
macro_f1_val = np.mean(f1_cls)

print("\n" + "="*45)
print(f"{'Metrica Globale':25s} | {'Valore':8s}")
print("="*45)
print(f"{'Media Macro ROC-AUC':25s} | {macro_roc_auc:8.4f}")
print(f"{'Media Micro ROC-AUC':25s} | {micro_roc_auc:8.4f}")
print(f"{'Macro PR-AUC':25s} | {macro_pr_auc:8.4f}")
print(f"{'Macro Precision':25s} | {macro_precision:8.4f}")
print(f"{'Macro Recall':25s} | {macro_recall:8.4f}")
print(f"{'Macro F1-Score':25s} | {macro_f1_val:8.4f}")
print("="*45)

# =====================================================
# SAVE REPORT
# =====================================================

out_path = os.path.join(
    BASE_DIR,
    "evaluation_ResNet50_v1.txt"
)

with open(out_path, "w") as f:
    f.write("ROC-AUC per class:\n")
    for c, r in zip(valid_classes, roc_scores):
        f.write(f"{c}: {r:.4f}\n")

    f.write("\nGLOBAL METRICS\n")
    f.write(f"Mean ROC-AUC: {mean_roc:.4f}\n")
    f.write(f"PR-AUC: {pr_auc:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")
    f.write(f"Micro F1: {micro_f1:.4f}\n")
    f.write(f"Samples F1: {samples_f1:.4f}\n")

print("\nSaved ✔")
print(out_path)