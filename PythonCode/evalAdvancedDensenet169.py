import os
import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    classification_report
)

from dataset import NIHChestXrayDataset, collate_fn

# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n========================")
print("DEVICE")
print("========================")
print(device)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode"

TEST_CSV = os.path.join(
    BASE_DIR,
    "test_split.csv"
)

IMAGE_ROOT = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "Advanced_DenseNet169_v1.pth"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "optimized_thresholds_DenseNet169_v1.json"
)

REPORT_PATH = os.path.join(
    BASE_DIR,
    "evaluation_DenseNet169_v1.txt"
)

# =========================================================
# CLASSES
# =========================================================

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

# =========================================================
# TEST DATA
# =========================================================

test_df = pd.read_csv(TEST_CSV)

IMAGE_SIZE = 320

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

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

# =========================================================
# MODEL
# =========================================================

print("\n========================")
print("LOADING MODEL")
print("========================")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = models.densenet169(weights=None)

model.classifier = torch.nn.Linear(
    model.classifier.in_features,
    num_classes
)

model.load_state_dict(checkpoint["model"])

model = model.to(device)
model.eval()

print("Model loaded ✔")

# =========================================================
# THRESHOLDS
# =========================================================

with open(THRESHOLD_PATH, "r") as f:
    threshold_dict = json.load(f)

print("Thresholds loaded ✔")

thresholds = np.array([
    threshold_dict[c]
    for c in classes
])

# =========================================================
# INFERENCE
# =========================================================

print("\n========================")
print("RUNNING INFERENCE")
print("========================")

all_probs = []
all_targets = []

with torch.no_grad():

    for images, labels in tqdm(test_loader):

        if images is None:
            continue

        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        all_probs.append(probs.cpu())
        all_targets.append(labels)

all_probs = torch.cat(all_probs).numpy()
all_targets = torch.cat(all_targets).numpy()

print("\nPredictions:", all_probs.shape)
print("Targets:", all_targets.shape)

# =========================================================
# ROC-AUC
# =========================================================

print("\n========================")
print("ROC-AUC PER CLASS")
print("========================")

roc_scores = []

for i, cls in enumerate(classes):

    score = roc_auc_score(
        all_targets[:, i],
        all_probs[:, i]
    )

    roc_scores.append(score)

    print(f"{cls:22s} {score:.4f}")

mean_roc = np.mean(roc_scores)

# =========================================================
# PR-AUC
# =========================================================

pr_auc = average_precision_score(
    all_targets,
    all_probs,
    average="macro"
)

# =========================================================
# APPLY THRESHOLDS
# =========================================================

pred_binary = (
    all_probs >= thresholds
).astype(int)

# =========================================================
# F1 SCORES
# =========================================================

macro_f1 = f1_score(
    all_targets,
    pred_binary,
    average="macro",
    zero_division=0
)

micro_f1 = f1_score(
    all_targets,
    pred_binary,
    average="micro",
    zero_division=0
)

samples_f1 = f1_score(
    all_targets,
    pred_binary,
    average="samples",
    zero_division=0
)

# =========================================================
# REPORT
# =========================================================

report = classification_report(
    all_targets,
    pred_binary,
    target_names=classes,
    digits=4,
    zero_division=0
)

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

# =========================================================
# SAVE REPORT
# =========================================================

with open(REPORT_PATH, "w") as f:

    f.write("DenseNet169 v1 Evaluation\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Mean ROC-AUC : {mean_roc:.6f}\n")
    f.write(f"PR-AUC       : {pr_auc:.6f}\n")
    f.write(f"Macro F1     : {macro_f1:.6f}\n")
    f.write(f"Micro F1     : {micro_f1:.6f}\n")
    f.write(f"Samples F1   : {samples_f1:.6f}\n\n")

    f.write(report)

print("\nSaved ✔")
print(REPORT_PATH)