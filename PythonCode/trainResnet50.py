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

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n========================")
print("DEVICE")
print("========================")
print(device)

# =========================
# PATHS
# =========================
BASE_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode"

TRAIN_CSV = os.path.join(BASE_DIR, "train_split.csv")
VAL_CSV   = os.path.join(BASE_DIR, "val_split.csv")
IMAGE_ROOT = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(
    CHECKPOINT_DIR,
    "Advanced_ResNet50_v1.pth"
)

# =========================
# DATA
# =========================
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

print("\n========================")
print("DATA")
print("========================")
print("Train:", len(train_df))
print("Val:", len(val_df))

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
# TRANSFORMS (più “ResNet-friendly”)
# =========================
IMAGE_SIZE = 320

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(8),
    transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.9, 1.1)),

    transforms.ColorJitter(brightness=0.1, contrast=0.1),

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

# =========================
# DATASETS
# =========================
train_dataset = NIHChestXrayDataset(train_df, IMAGE_ROOT, train_transform)
val_dataset   = NIHChestXrayDataset(val_df, IMAGE_ROOT, val_transform)

# =========================
# DATALOADERS
# =========================
train_loader = DataLoader(
    train_dataset,
    batch_size=32,   # ↑ ResNet regge meglio batch più grande
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn
)

print("\n========================")
print("DATALOADER")
print("========================")
print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))

# =========================
# POS WEIGHTS (same idea, ma stabile)
# =========================
print("\n========================")
print("COMPUTING POS WEIGHTS")
print("========================")

all_labels = []

temp_dataset = NIHChestXrayDataset(train_df, IMAGE_ROOT, None)

for i in range(len(temp_dataset)):
    _, label = temp_dataset[i]
    if label is not None:
        all_labels.append(label.numpy())

all_labels = np.array(all_labels)

pos = all_labels.sum(axis=0)
neg = len(all_labels) - pos

pos_weights = neg / (pos + 1e-6)

pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(device)

# =========================
# MODEL RESNET50
# =========================
print("\n========================")
print("LOADING RESNET50")
print("========================")

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# freeze backbone iniziale
for param in model.parameters():
    param.requires_grad = False

# unfreeze last layer block (più efficace di DenseNet freezing totale)
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, num_classes)
)

model = model.to(device)

print("Model loaded ✔")

# =========================
# LOSS
# =========================
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

# =========================
# OPTIMIZER (solo parametri trainabili)
# =========================
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)

# =========================
# SCHEDULER
# =========================
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,
    T_mult=2
)

# =========================
# AMP
# =========================
scaler = torch.amp.GradScaler("cuda")

# =========================
# TRAIN SETTINGS
# =========================
EPOCHS = 25
best_val_loss = float("inf")
early_counter = 0
patience = 6

# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):

    print("\n========================")
    print(f"EPOCH {epoch+1}/{EPOCHS}")
    print("========================")

    # unfreeze more after epoch 4
    if epoch == 4:
        print("\nUNFREEZING MORE LAYERS ✔")
        for param in model.layer3.parameters():
            param.requires_grad = True

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

    # =========================
    # VALIDATION
    # =========================
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

    # =========================
    # SAVE BEST
    # =========================
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save({
            "model": model.state_dict(),
            "classes": classes,
            "image_size": IMAGE_SIZE,
            "architecture": "ResNet50",
            "val_loss": val_loss,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        }, MODEL_SAVE_PATH)

        print("\n✔ BEST MODEL SAVED")
        early_counter = 0

    else:
        early_counter += 1
        print("\nEarly stopping:", early_counter)

    if early_counter >= patience:
        print("\n⛔ EARLY STOPPING")
        break

print("\nTRAINING COMPLETED ✔")
print("Saved at:", MODEL_SAVE_PATH)