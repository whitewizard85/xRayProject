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

print("\n========================")
print("DEVICE")
print("========================")
print(device)

# =========================================================
# PATHS
# =========================================================
BASE_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode"

TRAIN_CSV = os.path.join(BASE_DIR, "train_split.csv")
VAL_CSV   = os.path.join(BASE_DIR, "val_split.csv")

IMAGE_ROOT = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(
    CHECKPOINT_DIR,
    "Advanced_DenseNet169_v1.pth"
)

# =========================================================
# DATA
# =========================================================
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

print("\n========================")
print("DATA")
print("========================")
print("Train:", len(train_df))
print("Val:", len(val_df))

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
IMAGE_SIZE = 320

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(7),
    transforms.RandomAffine(0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.08, contrast=0.08),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =========================================================
# DATASETS
# =========================================================
train_dataset = NIHChestXrayDataset(
    train_df,
    image_root=IMAGE_ROOT,
    transform=train_transform
)

val_dataset = NIHChestXrayDataset(
    val_df,
    image_root=IMAGE_ROOT,
    transform=val_transform
)

# =========================================================
# DATALOADERS
# =========================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    collate_fn=collate_fn
)

print("\n========================")
print("DATALOADER")
print("========================")
print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))

# =========================================================
# POS WEIGHTS
# =========================================================
print("\n========================")
print("COMPUTING POS WEIGHTS")
print("========================")

all_labels = []

temp_dataset = NIHChestXrayDataset(
    train_df,
    image_root=IMAGE_ROOT,
    transform=None
)

for i in range(len(temp_dataset)):
    _, label = temp_dataset[i]
    if label is not None:
        all_labels.append(label.numpy())

all_labels = np.array(all_labels)

pos_counts = all_labels.sum(axis=0)
neg_counts = len(all_labels) - pos_counts

pos_weights = neg_counts / (pos_counts + 1e-6)

pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(device)

# =========================================================
# MODEL (DenseNet169)
# =========================================================
print("\n========================")
print("LOADING MODEL")
print("========================")

model = models.densenet169(
    weights=models.DenseNet169_Weights.IMAGENET1K_V1
)

for param in model.features.parameters():
    param.requires_grad = False

model.classifier = nn.Linear(
    model.classifier.in_features,
    num_classes
)

model = model.to(device)

print("Model loaded ✔")

# =========================================================
# LOSS
# =========================================================
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

# =========================================================
# OPTIMIZER
# =========================================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=1e-4
)

# =========================================================
# SCHEDULER
# =========================================================
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,
    T_mult=2
)

# =========================================================
# AMP
# =========================================================
scaler = torch.amp.GradScaler("cuda")

# =========================================================
# TRAIN SETTINGS
# =========================================================
EPOCHS = 30
best_val_loss = float("inf")
early_counter = 0
early_patience = 7

best_weights = copy.deepcopy(model.state_dict())

# =========================================================
# TRAIN LOOP
# =========================================================
for epoch in range(EPOCHS):

    print("\n========================")
    print(f"EPOCH {epoch+1}/{EPOCHS}")
    print("========================")

    if epoch == 3:
        print("\nUNFREEZING BACKBONE ✔")
        for p in model.features.parameters():
            p.requires_grad = True

    # ---------------- TRAIN ----------------
    model.train()
    train_loss = 0

    for images, labels in tqdm(train_loader):

        if images is None:
            continue

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader):

            if images is None:
                continue

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

    val_loss /= len(val_loader)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # ---------------- METRICS ----------------
    try:
        roc_auc = roc_auc_score(all_targets, all_preds, average="macro")
        pr_auc = average_precision_score(all_targets, all_preds, average="macro")
    except:
        roc_auc = 0
        pr_auc = 0

    scheduler.step(epoch + 1)

    print("\n========================")
    print("RESULTS")
    print("========================")
    print("Train Loss:", round(train_loss, 4))
    print("Val Loss  :", round(val_loss, 4))
    print("ROC-AUC   :", round(roc_auc, 4))
    print("PR-AUC    :", round(pr_auc, 4))

    # ---------------- SAVE BEST ----------------
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_weights = copy.deepcopy(model.state_dict())

        torch.save({
            "model": best_weights,
            "classes": classes,
            "image_size": IMAGE_SIZE,
            "architecture": "DenseNet169",
            "val_loss": val_loss,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        }, MODEL_SAVE_PATH)

        print("\n✔ BEST MODEL SAVED")
        early_counter = 0

    else:
        early_counter += 1
        print("\nEarly stopping:", early_counter)

    if early_counter >= early_patience:
        print("\n⛔ EARLY STOPPING")
        break

print("\nTRAINING COMPLETED ✔")
print("Saved at:", MODEL_SAVE_PATH)