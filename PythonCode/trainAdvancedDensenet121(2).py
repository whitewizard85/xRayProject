import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from dataset import NIHDataset, collate_fn
from preprocessing import train_transform, val_transform

# =========================
# BASE PATH (FIX DEFINITIVO)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_csv = os.path.join(BASE_DIR, "train_split.csv")
val_csv   = os.path.join(BASE_DIR, "val_split.csv")

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDEVICE:", device)

# =========================
# LOAD DATA
# =========================
train_df = pd.read_csv(train_csv)
val_df   = pd.read_csv(val_csv)

train_loader = DataLoader(
    NIHDataset(train_df, transform=train_transform),
    batch_size=32,
    shuffle=True,
    num_workers=2,   # più stabile su VM
    pin_memory=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    NIHDataset(val_df, transform=val_transform),
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn
)

# =========================
# CLASSES (15 WITH NO FINDING)
# =========================
classes = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax",
    "Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia","No Finding"
]

num_classes = len(classes)

# =========================
# MODEL
# =========================
model = models.densenet121(weights="IMAGENET1K_V1")
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model = model.to(device)

print("\nModel loaded ✔")

# =========================
# FOCAL LOSS (STABILE VERSION)
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        bce = self.bce(inputs, targets)
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()

criterion = FocalLoss()

# =========================
# OPTIMIZER / SCHEDULER
# =========================
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

scaler = torch.cuda.amp.GradScaler()

# =========================
# TRAIN SETTINGS
# =========================
EPOCHS = 20
best_val_loss = float("inf")
early_stop = 5
counter = 0

os.makedirs(os.path.join(BASE_DIR, "checkpoints"), exist_ok=True)

# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for images, labels in loop:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    train_loss /= len(train_loader)

    # =========================
    # VALIDATION
    # =========================
    model.eval()
    val_loss = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            all_preds.append(torch.sigmoid(outputs).cpu())
            all_targets.append(labels.cpu())

    val_loss /= len(val_loader)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # =========================
    # METRICS
    # =========================
    try:
        roc_auc = roc_auc_score(all_targets, all_preds, average="macro")
        pr_auc = average_precision_score(all_targets, all_preds, average="macro")
    except:
        roc_auc = 0
        pr_auc = 0

    scheduler.step(val_loss)

    lr = optimizer.param_groups[0]["lr"]

    print("\n========================")
    print(f"Epoch {epoch+1}")
    print("========================")
    print("Train Loss:", round(train_loss, 4))
    print("Val Loss:", round(val_loss, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))
    print("LR:", lr)

    # =========================
    # SAVE BEST MODEL
    # =========================
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        counter = 0

        torch.save({
            "model": model.state_dict(),
            "classes": classes
        }, os.path.join(BASE_DIR, "checkpoints", "Advanced_DenseNet121.pth"))

        print("✔ Model saved")

    else:
        counter += 1

    if counter >= early_stop:
        print("\nEarly stopping triggered ⛔")
        break