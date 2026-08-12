import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torchvision import models, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
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

MODEL_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "EfficientNetB0_v1.pth"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "optimized_thresholds_EfficientNetB0.json"
)

TEST_CSV = os.path.join(
    BASE_DIR,
    "test_split.csv"
)

IMAGE_ROOT = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

# =====================================================
# CLASSES
# =====================================================

classes = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "No Finding"
]

num_classes = len(classes)

# =====================================================
# TRANSFORM
# =====================================================

IMAGE_SIZE = 224

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# DATASET
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
# MODEL
# =====================================================

print("\n========================")
print("LOADING MODEL")
print("========================")

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = models.efficientnet_b0()

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model.load_state_dict(checkpoint["model"])

model.to(device)
model.eval()

print("Model loaded ✔")

# =====================================================
# LOAD THRESHOLDS
# =====================================================

with open(THRESHOLD_PATH, "r") as f:
    thresholds_dict = json.load(f)

thresholds = np.array([thresholds_dict[c] for c in classes])

print("Thresholds loaded ✔")

# =====================================================
# INFERENCE
# =====================================================

all_probs = []
all_targets = []

print("\n========================")
print("RUNNING INFERENCE")
print("========================")

with torch.no_grad():

    for images, labels in tqdm(test_loader):

        if images is None:
            continue

        images = images.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)

        all_probs.append(probs.cpu().numpy())
        all_targets.append(labels.numpy())

all_probs = np.vstack(all_probs)
all_targets = np.vstack(all_targets)

print("\nPredictions:", all_probs.shape)
print("Targets:", all_targets.shape)

# =====================================================
# ROC AUC
# =====================================================

print("\n========================")
print("ROC-AUC PER CLASS")
print("========================")

roc_scores = []

for i, cls in enumerate(classes):

    auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
    roc_scores.append(auc)

    print(f"{cls:22s} {auc:.4f}")

mean_roc = np.mean(roc_scores)

# =====================================================
# PR AUC
# =====================================================

pr_auc = average_precision_score(
    all_targets,
    all_probs,
    average="macro"
)

# =====================================================
# APPLY THRESHOLDS
# =====================================================

pred_binary = (all_probs >= thresholds).astype(int)

# =====================================================
# METRICS
# =====================================================

report = classification_report(
    all_targets,
    pred_binary,
    target_names=classes,
    digits=4
)

macro_f1 = f1_score(all_targets, pred_binary, average="macro")
micro_f1 = f1_score(all_targets, pred_binary, average="micro")
samples_f1 = f1_score(all_targets, pred_binary, average="samples")

# =====================================================
# RESULTS
# =====================================================

print("\n========================")
print("GLOBAL RESULTS")
print("========================")

print("Mean ROC-AUC :", round(mean_roc, 4))
print("PR-AUC       :", round(pr_auc, 4))
print("Macro F1     :", round(macro_f1, 4))
print("Micro F1     :", round(micro_f1, 4))
print("Samples F1   :", round(samples_f1, 4))

print("\n")
print(report)

# =====================================================
# SAVE REPORT
# =====================================================

report_path = os.path.join(
    BASE_DIR,
    "evaluation_EfficientNetB0.txt"
)

with open(report_path, "w") as f:

    f.write("EfficientNetB0 Evaluation\n\n")
    f.write(f"Mean ROC-AUC: {mean_roc:.4f}\n")
    f.write(f"PR-AUC: {pr_auc:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")
    f.write(f"Micro F1: {micro_f1:.4f}\n")
    f.write(f"Samples F1: {samples_f1:.4f}\n\n")
    f.write(report)

print("\nSaved ✔")
print(report_path)