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
    f1_score,
    precision_recall_curve,
    auc
)

from dataset import NIHChestXrayDataset

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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
    "Advanced_DenseNet121_v2.pth"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "optimized_thresholds_v2.json"
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

IMAGE_SIZE = 320

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =====================================================
# LOAD TEST
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
    pin_memory=True
)

print("\nTest samples:", len(test_dataset))
print("Test batches:", len(test_loader))

# =====================================================
# LOAD MODEL
# =====================================================

print("\n========================")
print("LOADING MODEL")
print("========================")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = models.densenet121()

model.classifier = nn.Linear(
    model.classifier.in_features,
    num_classes
)

model.load_state_dict(
    checkpoint["model"]
)

model.to(device)
model.eval()

print("Model loaded ✔")

# =====================================================
# LOAD THRESHOLDS
# =====================================================

with open(THRESHOLD_PATH, "r") as f:
    thresholds_dict = json.load(f)

thresholds = np.array(
    [thresholds_dict[c] for c in classes]
)

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

        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        all_probs.append(
            probs.cpu().numpy()
        )

        all_targets.append(
            labels.numpy()
        )

all_probs = np.vstack(all_probs)
all_targets = np.vstack(all_targets)

print("\nPredictions:", all_probs.shape)
print("Targets:", all_targets.shape)

# =====================================================
# GLOBAL METRICS COMPUTATION (Macro/Micro ROC-AUC, PR-AUC, Precision, Recall, F1)
# =====================================================

# 1. ROC-AUC
macro_auc = roc_auc_score(all_targets, all_probs, average="macro")
micro_auc = roc_auc_score(all_targets, all_probs, average="micro")

# 2. Macro PR-AUC
pr_auc_list = []
for j in range(num_classes):
    precision_c, recall_c, _ = precision_recall_curve(all_targets[:, j], all_probs[:, j])
    pr_auc_list.append(auc(recall_c, precision_c))
macro_pr_auc = np.mean(pr_auc_list)

# 3. Apply Thresholds for Binary Predictions
pred_binary = (
    all_probs >= thresholds
).astype(int)

# 4. Classification Report & Averages
report_dict = classification_report(
    all_targets,
    pred_binary,
    target_names=classes,
    output_dict=True,
    zero_division=0
)

macro_precision = report_dict['macro avg']['precision']
macro_recall = report_dict['macro avg']['recall']
macro_f1 = report_dict['macro avg']['f1-score']
micro_f1 = report_dict['micro avg']['f1-score']
samples_f1 = report_dict['samples avg']['f1-score']

report_str = classification_report(
    all_targets,
    pred_binary,
    target_names=classes,
    digits=4,
    zero_division=0
)

# =====================================================
# RESULTS SUMMARY
# =====================================================

print("\n" + "="*50)
print("FINAL METRICS SUMMARY (TEST SET)")
print("="*50)
print(f"Macro ROC-AUC   : {macro_auc:.4f}")
print(f"Micro ROC-AUC   : {micro_auc:.4f}")
print(f"Macro PR-AUC    : {macro_pr_auc:.4f}")
print(f"Macro Precision : {macro_precision:.4f}")
print(f"Macro Recall    : {macro_recall:.4f}")
print(f"Macro F1-Score  : {macro_f1:.4f}")
print(f"Micro F1-Score  : {micro_f1:.4f}")
print(f"Samples F1-Score: {samples_f1:.4f}")

print("\n" + "="*50)
print("CLASSIFICATION REPORT DETTAGLIATO:")
print("="*50)
print(report_str)

# =====================================================
# TABELLA UNIFICATA PER EXCEL (Classe | ROC-AUC | Precision | Recall | F1-Score | Support)
# =====================================================

print("\n" + "="*50)
print("TABELLA UNIFICATA PER EXCEL (Classe | ROC-AUC | Precision | Recall | F1-Score | Support):")
print("="*50)
print("Classe\tROC-AUC\tPrecision\tRecall\tF1-Score\tSupport")
for j, c in enumerate(classes):
    try:
        auc_c = roc_auc_score(all_targets[:, j], all_probs[:, j])
    except ValueError:
        auc_c = 0.0
    p = report_dict[c]['precision']
    r = report_dict[c]['recall']
    f = report_dict[c]['f1-score']
    supp = report_dict[c]['support']
    print(f"{c}\t{auc_c:.4f}\t{p:.2f}\t{r:.2f}\t{f:.2f}\t{supp}")

# =====================================================
# SAVE REPORT
# =====================================================

report_path = os.path.join(
    BASE_DIR,
    "evaluation_DenseNet121_v2.txt"
)

with open(report_path, "w") as f:
    f.write("DenseNet121_v2 Evaluation\n\n")
    f.write(f"Macro ROC-AUC: {macro_auc:.4f}\n")
    f.write(f"Micro ROC-AUC: {micro_auc:.4f}\n")
    f.write(f"Macro PR-AUC: {macro_pr_auc:.4f}\n")
    f.write(f"Macro Precision: {macro_precision:.4f}\n")
    f.write(f"Macro Recall: {macro_recall:.4f}\n")
    f.write(f"Macro F1-Score: {macro_f1:.4f}\n")
    f.write(f"Micro F1-Score: {micro_f1:.4f}\n")
    f.write(f"Samples F1-Score: {samples_f1:.4f}\n\n")
    f.write(report_str)

print("\nSaved ✔")
print(report_path)