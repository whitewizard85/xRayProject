import os
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from dataset import NIHChestXrayDataset, collate_fn

# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)

# =========================================================
# PATHS
# =========================================================
BASE_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode"

TRAIN_CSV = os.path.join(BASE_DIR, "train_split.csv")
VAL_CSV   = os.path.join(BASE_DIR, "val_split.csv")

IMAGE_ROOT = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "EfficientNetB0_v1.pth")

# =========================================================
# DATA
# =========================================================
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

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
# TRANSFORMS (IMAGENET STANDARD)
# =========================================================
IMAGE_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(7),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# =========================================================
# DATASETS
# =========================================================
train_dataset = NIHChestXrayDataset(train_df, IMAGE_ROOT, train_transform)
val_dataset   = NIHChestXrayDataset(val_df, IMAGE_ROOT, val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn
)

# =========================================================
# POS WEIGHTS
# =========================================================
print("Computing pos weights...")

all_labels = []
temp = NIHChestXrayDataset(train_df, IMAGE_ROOT, None)

for i in range(len(temp)):
    _, y = temp[i]
    if y is not None:
        all_labels.append(y.numpy())

all_labels = np.array(all_labels)

pos = all_labels.sum(axis=0)
neg = len(all_labels) - pos

pos_weights = torch.tensor(neg / (pos + 1e-6),
                           dtype=torch.float32).to(device)

# =========================================================
# MODEL (EfficientNet-B0)
# =========================================================
print("Loading EfficientNet-B0...")

model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
)

# freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(device)

# =========================================================
# LOSS / OPTIMIZER
# =========================================================
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=1e-4,
                              weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=15
)

scaler = torch.amp.GradScaler("cuda")

# =========================================================
# TRAINING
# =========================================================
EPOCHS = 20
best_val = float("inf")
patience = 6
counter = 0

for epoch in range(EPOCHS):

    print(f"\nEPOCH {epoch+1}/{EPOCHS}")

    # UNFREEZE
    if epoch == 3:
        print("Unfreezing backbone")
        for p in model.features.parameters():
            p.requires_grad = True

    # TRAIN
    model.train()
    train_loss = 0

    for x, y in tqdm(train_loader):

        if x is None:
            continue

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # VAL
    model.eval()
    val_loss = 0

    preds, targets = [], []

    with torch.no_grad():
        for x, y in val_loader:

            if x is None:
                continue

            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

            val_loss += loss.item()

            preds.append(torch.sigmoid(out).cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    try:
        roc = roc_auc_score(targets, preds, average="macro")
        pr = average_precision_score(targets, preds, average="macro")
    except:
        roc, pr = 0, 0

    scheduler.step()

    print("Train Loss:", train_loss)
    print("Val Loss:", val_loss / len(val_loader))
    print("ROC-AUC:", roc)
    print("PR-AUC:", pr)

    if val_loss < best_val:
        best_val = val_loss
        counter = 0

        torch.save({
            "model": model.state_dict(),
            "classes": classes
        }, MODEL_SAVE_PATH)

        print("✔ SAVED BEST MODEL")

    else:
        counter += 1
        print("Early stopping:", counter)

    if counter >= patience:
        print("STOP EARLY")
        break

print("DONE ✔")