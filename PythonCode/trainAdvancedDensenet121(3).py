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
    "Advanced_DenseNet121_v2.pth"
)

# =========================================================
# LOAD DATAFRAMES
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
# ADVANCED TRANSFORMS
# =========================================================

IMAGE_SIZE = 320

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(7),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05)
    ),

    transforms.ColorJitter(
        brightness=0.08,
        contrast=0.08
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
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
# POSITIVE WEIGHTS
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

positive_counts = all_labels.sum(axis=0)
negative_counts = len(all_labels) - positive_counts

pos_weights = negative_counts / (positive_counts + 1e-6)

print("\nPOS WEIGHTS:")
for c, w in zip(classes, pos_weights):
    print(f"{c:25s} {w:.2f}")

pos_weights = torch.tensor(
    pos_weights,
    dtype=torch.float32
).to(device)

# =========================================================
# MODEL
# =========================================================

print("\n========================")
print("LOADING MODEL")
print("========================")

model = models.densenet121(
    weights=models.DenseNet121_Weights.IMAGENET1K_V1
)

# freeze first layers initially
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

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weights
)

# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

# =========================================================
# COSINE SCHEDULER
# =========================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=20
)

# =========================================================
# MIXED PRECISION
# =========================================================

scaler = torch.amp.GradScaler("cuda")

# =========================================================
# TRAIN SETTINGS
# =========================================================

EPOCHS = 20

best_val_loss = float("inf")

early_stopping_patience = 5
early_counter = 0

best_model_weights = copy.deepcopy(model.state_dict())

# =========================================================
# TRAIN LOOP
# =========================================================

for epoch in range(EPOCHS):

    print("\n================================================")
    print(f"EPOCH {epoch+1}/{EPOCHS}")
    print("================================================")

    # =====================================================
    # UNFREEZE BACKBONE AFTER 2 EPOCHS
    # =====================================================

    if epoch == 2:

        print("\nUNFREEZING BACKBONE ✔")

        for param in model.features.parameters():
            param.requires_grad = True

    # =====================================================
    # TRAIN
    # =====================================================

    model.train()

    train_loss = 0

    train_loop = tqdm(train_loader)

    for images, labels in train_loop:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):

            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)

        scaler.update()

        train_loss += loss.item()

        train_loop.set_postfix(
            loss=round(loss.item(), 4)
        )

    train_loss /= len(train_loader)

    # =====================================================
    # VALIDATION
    # =====================================================

    model.eval()

    val_loss = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():

        for images, labels in tqdm(val_loader):

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

    # =====================================================
    # METRICS
    # =====================================================

    try:

        roc_auc = roc_auc_score(
            all_targets,
            all_preds,
            average="macro"
        )

        pr_auc = average_precision_score(
            all_targets,
            all_preds,
            average="macro"
        )

    except:

        roc_auc = 0
        pr_auc = 0

    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]

    # =====================================================
    # REPORT
    # =====================================================

    print("\n========================")
    print("RESULTS")
    print("========================")

    print("Train Loss :", round(train_loss, 4))
    print("Val Loss   :", round(val_loss, 4))
    print("ROC-AUC    :", round(roc_auc, 4))
    print("PR-AUC     :", round(pr_auc, 4))
    print("LR         :", current_lr)

    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save({

            "model": best_model_weights,
            "classes": classes,
            "image_size": IMAGE_SIZE,
            "val_loss": val_loss,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc

        }, MODEL_SAVE_PATH)

        print("\n✔ BEST MODEL SAVED")
        print(MODEL_SAVE_PATH)

        early_counter = 0

    else:

        early_counter += 1

        print(f"\nEarly stopping counter: {early_counter}")

    # =====================================================
    # EARLY STOPPING
    # =====================================================

    if early_counter >= early_stopping_patience:

        print("\n⛔ EARLY STOPPING TRIGGERED")
        break

# =========================================================
# TRAINING COMPLETED
# =========================================================

print("\n================================================")
print("TRAINING COMPLETED ✔")
print("================================================")

print("\nBest model saved at:")
print(MODEL_SAVE_PATH)