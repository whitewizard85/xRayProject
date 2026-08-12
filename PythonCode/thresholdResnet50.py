import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from tqdm import tqdm
from sklearn.metrics import f1_score

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

VAL_CSV = os.path.join(BASE_DIR, "val_split.csv")

IMAGE_ROOT = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "Advanced_ResNet50_v1.pth"
)

THRESHOLD_SAVE = os.path.join(
    BASE_DIR,
    "optimized_thresholds_ResNet50_v1.json"
)

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
# DATA
# =====================================================

val_df = pd.read_csv(VAL_CSV)

IMAGE_SIZE = 320

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

val_dataset = NIHChestXrayDataset(
    val_df,
    image_root=IMAGE_ROOT,
    transform=transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    collate_fn=collate_fn
)

print("\nValidation samples:", len(val_dataset))
print("Validation batches:", len(val_loader))

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
# INFERENCE
# =====================================================

print("\n========================")
print("GENERATING PREDICTIONS")
print("========================")

all_preds = []
all_targets = []

with torch.no_grad():

    for images, labels in tqdm(val_loader):

        if images is None:
            continue

        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        all_preds.append(
            probs.cpu().numpy()
        )

        all_targets.append(
            labels.numpy()
        )

preds = np.concatenate(all_preds)
targets = np.concatenate(all_targets)

print("\nPredictions shape:", preds.shape)
print("Targets shape:", targets.shape)

# =====================================================
# THRESHOLD SEARCH
# =====================================================

print("\n========================")
print("THRESHOLD SEARCH")
print("========================")

thresholds = {}

for i, cls in enumerate(classes):

    best_thr = 0.5
    best_f1 = 0

    y_true = targets[:, i]
    y_prob = preds[:, i]

    for thr in np.arange(0.05, 1.00, 0.05):

        y_pred = (y_prob >= thr).astype(int)

        score = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_thr = float(thr)

    thresholds[cls] = best_thr

    print(
        f"{cls:25s} | "
        f"Threshold: {best_thr:.2f} | "
        f"F1: {best_f1:.4f}"
    )

# =====================================================
# SAVE
# =====================================================

with open(THRESHOLD_SAVE, "w") as f:
    json.dump(
        thresholds,
        f,
        indent=4
    )

print("\n========================")
print("THRESHOLDS SAVED")
print("========================")
print(THRESHOLD_SAVE)

print("\nDONE ✔")