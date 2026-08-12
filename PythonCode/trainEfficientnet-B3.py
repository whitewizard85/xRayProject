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

# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n========================")
print("DEVICE")
print("========================")
print(device)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode"

VAL_CSV = os.path.join(BASE_DIR, "val_split.csv")

IMAGE_ROOT = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "EfficientNetB3_ASL.pth"
)

THRESHOLD_SAVE_PATH = os.path.join(
    BASE_DIR,
    "optimized_thresholds_EfficientNetB3.json"
)

# =========================================================
# CLASSES
# =========================================================

classes = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax",
    "Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia","No Finding"
]

num_classes = len(classes)

# =========================================================
# TRANSFORMS
# =========================================================

IMAGE_SIZE = 300

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# DATA
# =========================================================

val_df = pd.read_csv(VAL_CSV)

val_dataset = NIHChestXrayDataset(
    val_df,
    image_root=IMAGE_ROOT,
    transform=val_transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn
)

print("\nValidation samples:", len(val_dataset))
print("Validation batches:", len(val_loader))

# =========================================================
# MODEL
# =========================================================

print("\n========================")
print("LOADING MODEL")
print("========================")

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = models.efficientnet_b3(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model.load_state_dict(checkpoint["model"])
model = model.to(device)
model.eval()

print("Model loaded ✔")

# =========================================================
# INFERENCE
# =========================================================

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

        all_preds.append(probs.cpu())
        all_targets.append(labels)

all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

print("\nPredictions shape:", all_preds.shape)
print("Targets shape:", all_targets.shape)

# =========================================================
# THRESHOLD SEARCH
# =========================================================

print("\n========================")
print("THRESHOLD SEARCH")
print("========================")

best_thresholds = {}

search_space = np.arange(0.05, 0.96, 0.05)

for class_idx, class_name in enumerate(classes):

    y_true = all_targets[:, class_idx]
    y_prob = all_preds[:, class_idx]

    # sicurezza: evita classi senza positivi o negativi
    if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
        best_thresholds[class_name] = 0.5
        print(f"{class_name:25s} | SKIPPED (degenerate)")
        continue

    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in search_space:

        y_pred = (y_prob >= threshold).astype(int)

        score = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    best_thresholds[class_name] = best_threshold

    print(
        f"{class_name:25s} "
        f"| Threshold: {best_threshold:.2f} "
        f"| F1: {best_f1:.4f}"
    )

# =========================================================
# SAVE JSON
# =========================================================

with open(THRESHOLD_SAVE_PATH, "w") as f:
    json.dump(best_thresholds, f, indent=4)

print("\n========================")
print("THRESHOLDS SAVED")
print("========================")
print(THRESHOLD_SAVE_PATH)

print("\nDONE ✔")