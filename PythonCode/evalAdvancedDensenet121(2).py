import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from dataset import NIHChestXrayDataset, encode_labels
from preprocessing import val_transform

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDEVICE:", device)

# =========================
# PATHS
# =========================
image_root = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
test_df = pd.read_csv(test_csv)

# =========================
# DATALOADER
# =========================
test_loader = DataLoader(
    NIHChestXrayDataset(test_df, image_root, transform=val_transform),
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("\nTest batches:", len(test_loader))

# =========================
# MODEL
# =========================
num_classes = 15

model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

# =========================
# LOAD CHECKPOINT
# =========================
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/checkpoints/Advanced_DenseNet121.pth"

print("\nLOADING MODEL")
print(checkpoint_path)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model"])

model = model.to(device)
model.eval()

print("Model loaded ✔")

# =========================
# EVALUATION
# =========================
all_preds = []
all_targets = []

print("\nEVALUATING...")

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.sigmoid(outputs)

        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())

all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

print("\nPredictions shape:", all_preds.shape)
print("Targets shape:", all_targets.shape)

# =========================
# METRICS
# =========================
classes = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax",
    "Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia","No Finding"
]

print("\nROC-AUC PER CLASS")
print("========================")

for i, c in enumerate(classes):
    try:
        auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
    except:
        auc = 0.0
    print(f"{c:20s} | {auc:.4f}")

# =========================
# GLOBAL METRICS
# =========================
try:
    mean_auc = roc_auc_score(all_targets, all_preds, average="macro")
    map_score = average_precision_score(all_targets, all_preds, average="macro")
except:
    mean_auc = 0
    map_score = 0

print("\n========================")
print("GLOBAL METRICS")
print("========================")
print("Mean ROC-AUC:", round(mean_auc, 4))
print("mAP (PR-AUC):", round(map_score, 4))