import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from dataset import NIHChestXrayDataset
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
threshold_path = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/optimized_thresholds.json"
model_path = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/checkpoints/Advanced_DenseNet121.pth"

# =========================
# LOAD DATA
# =========================
test_df = pd.read_csv(test_csv)

test_loader = DataLoader(
    NIHChestXrayDataset(test_df, image_root, transform=val_transform),
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("\nTest batches:", len(test_loader))

# =========================
# CLASSES
# =========================
classes = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax",
    "Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia","No Finding"
]

num_classes = len(classes)

# =========================
# MODEL
# =========================
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model = model.to(device)
model.eval()

print("\nModel loaded ✔")

# =========================
# LOAD THRESHOLDS
# =========================
with open(threshold_path, "r") as f:
    thresholds = json.load(f)

print("\nThresholds loaded ✔")

# =========================
# PREDICTIONS
# =========================
all_probs = []
all_targets = []

print("\nGENERATING PREDICTIONS...")

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)

        all_probs.append(probs.cpu())
        all_targets.append(labels.cpu())

all_probs = torch.cat(all_probs).numpy()
all_targets = torch.cat(all_targets).numpy()

# =========================
# APPLY THRESHOLDS
# =========================
print("\nAPPLYING THRESHOLDS...")

preds_bin = np.zeros_like(all_probs)

for i, cls in enumerate(classes):
    t = thresholds.get(cls, 0.5)
    preds_bin[:, i] = (all_probs[:, i] >= t).astype(int)

# =========================
# REPORT
# =========================
print("\n========================")
print("CLASSIFICATION REPORT (THRESHOLDED)")
print("========================")

report = classification_report(
    all_targets,
    preds_bin,
    target_names=classes,
    zero_division=0
)

print(report)