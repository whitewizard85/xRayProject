import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, classification_report

from dataset import NIHChestXrayDataset
from preprocessing import val_transform


# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDEVICE:", device)


# =========================
# PATH FIX DEFINITIVO
# =========================

BASE_DIR = "/home/gpuvm/Desktop/Luca Migliaccio"

MODEL_PATH = os.path.join(BASE_DIR, "best_densenet121.pth")
TEST_CSV = os.path.join(BASE_DIR, "PythonCode", "test_split.csv")


# =========================
# LOAD DATA
# =========================

test_df = pd.read_csv(TEST_CSV)

test_dataset = NIHChestXrayDataset(test_df, transform=val_transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("\nTest samples:", len(test_dataset))


# =========================
# MODEL
# =========================

model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 14)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model = model.to(device)
model.eval()

print("\nModel loaded ✔")


# =========================
# CLASS NAMES
# =========================

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]


# =========================
# INFERENCE
# =========================

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in tqdm(test_loader, desc="Evaluating"):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)

        y_true.append(labels.cpu().numpy())
        y_pred.append(probs.cpu().numpy())


y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)


# =========================
# ROC-AUC PER CLASS
# =========================

print("\n========================")
print("ROC-AUC PER CLASS")
print("========================")

auc_scores = []

for i, cls in enumerate(classes):

    try:
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        auc_scores.append(auc)
        print(f"{cls}: {auc:.4f}")
    except:
        print(f"{cls}: NOT COMPUTABLE")


# =========================
# MEAN AUC
# =========================

print("\n========================")
print("MEAN ROC-AUC")
print("========================")

print("Mean AUC:", round(np.mean(auc_scores), 4))


# =========================
# CLASSIFICATION REPORT
# =========================

y_pred_bin = (y_pred > 0.5).astype(int)

print("\n========================")
print("CLASSIFICATION REPORT")
print("========================")

print(classification_report(y_true, y_pred_bin, target_names=classes))