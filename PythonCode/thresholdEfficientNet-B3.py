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
    "EfficientNetB3_ASL",
    "data.pkl"
)

THRESHOLD_SAVE_PATH = os.path.join(
    BASE_DIR,
    "optimized_thresholds_EfficientNetB3_ASL.json"
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

num_classes = len(classes)

# =====================================================
# TRANSFORMS
# =====================================================

IMAGE_SIZE = 300

val_transform = transforms.Compose([
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

val_df = pd.read_csv(VAL_CSV)

val_dataset = NIHChestXrayDataset(
    val_df,
    IMAGE_ROOT,
    val_transform
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
# MODEL (EfficientNet-B3)
# =====================================================

print("\n========================")
print("LOADING MODEL")
print("========================")

model = models.efficientnet_b3(weights="IMAGENET1K_V1")

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

# =====================================================
# LOAD CHECKPOINT (FIXED PYTORCH 2.6+)
# =====================================================

ckpt = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False   # <<< FIX IMPORTANTE
)

# estrazione state_dict robusta
if isinstance(ckpt, dict):
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
else:
    state_dict = ckpt

# fix chiavi (Lightning / wrapper safe)
new_state_dict = {}
for k, v in state_dict.items():
    new_k = k.replace("model.", "").replace("module.", "")
    new_state_dict[new_k] = v

model.load_state_dict(new_state_dict, strict=False)

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

        all_preds.append(probs.cpu())
        all_targets.append(labels.cpu())

all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

print("\nPred shape:", all_preds.shape)
print("Target shape:", all_targets.shape)

# =====================================================
# THRESHOLD SEARCH
# =====================================================

print("\n========================")
print("THRESHOLD SEARCH")
print("========================")

search_space = np.arange(0.05, 0.96, 0.05)

best_thresholds = {}

for i, cls in enumerate(classes):

    y_true = all_targets[:, i]
    y_prob = all_preds[:, i]

    best_f1 = 0
    best_t = 0.5

    for t in search_space:

        y_pred = (y_prob >= t).astype(int)

        score = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_t = float(t)

    best_thresholds[cls] = best_t

    print(f"{cls:22s} | Thr: {best_t:.2f} | F1: {best_f1:.4f}")

# =====================================================
# SAVE
# =====================================================

with open(THRESHOLD_SAVE_PATH, "w") as f:
    json.dump(best_thresholds, f, indent=4)

print("\n========================")
print("SAVED")
print("========================")
print(THRESHOLD_SAVE_PATH)

print("\nDONE ✔")