import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchxrayvision as xrv
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

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

IMG_SIZE = 224

# =========================================================
# LABELS
# =========================================================
classes = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax",
    "Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia","No Finding"
]

num_classes = len(classes)

def encode_labels(label_str):
    y = np.zeros(num_classes, dtype=np.float32)
    for i, c in enumerate(classes):
        if c in label_str:
            y[i] = 1.0
    return torch.tensor(y)

# =========================================================
# IMAGE FINDER
# =========================================================
def find_image(root, name):
    for i in range(1, 13):
        p = os.path.join(root, f"images_{i:03d}", "images", name)
        if os.path.exists(p):
            return p
    return None

# =========================================================
# DATASET (SAFE + CONSISTENT)
# =========================================================
class XRVDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = find_image(IMAGE_ROOT, row["Image Index"])
        if path is None:
            raise ValueError(f"Missing image: {row['Image Index']}")

        img = xrv.utils.load_image(path).astype(np.float32)
        img = np.nan_to_num(img)

        # normalize per-image (safe baseline)
        img = img - img.min()
        img = img / (img.max() + 1e-6)

        img = torch.from_numpy(img).float()

        # FORCE SHAPE (1, H, W)
        if img.ndim == 2:
            img = img.unsqueeze(0)

        # resize using interpolate
        img = img.unsqueeze(0)  # (1,1,H,W)

        img = nn.functional.interpolate(
            img,
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False
        )

        img = img.squeeze(0)  # (1,224,224)

        # scale to XRV expected range
        img = img * 255.0

        label = encode_labels(row["Finding Labels"])

        return img, label

# =========================================================
# DATA
# =========================================================
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

train_loader = DataLoader(
    XRVDataset(train_df),
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    XRVDataset(val_df),
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# =========================================================
# MODEL (IMPORTANT FIX)
# =========================================================
print("Loading XRV DenseNet...")

model = xrv.models.DenseNet(weights="densenet121-res224-all")

# ⚠️ IMPORTANT: disable internal XRV post-processing logic
model.op_threshs = None

# replace classifier safely
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

model = model.to(device)

# =========================================================
# LOSS
# =========================================================
print("Computing pos_weights...")

labels = np.stack([
    encode_labels(train_df.iloc[i]["Finding Labels"]).numpy()
    for i in range(len(train_df))
])

pos_weights = (len(labels) - labels.sum(axis=0)) / (labels.sum(axis=0) + 1e-6)
pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

# =========================================================
# OPTIMIZER
# =========================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler()

# =========================================================
# TRAIN LOOP
# =========================================================
EPOCHS = 10

for epoch in range(EPOCHS):

    print(f"\nEPOCH {epoch+1}/{EPOCHS}")

    # TRAIN
    model.train()
    train_loss = 0

    for x, y in tqdm(train_loader):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # VALIDATION
    model.eval()

    val_loss = 0
    preds, targets = [], []

    with torch.no_grad():
        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = criterion(out, y)

            val_loss += loss.item()

            preds.append(torch.sigmoid(out).cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    try:
        roc = roc_auc_score(targets, preds, average="macro")
        pr  = average_precision_score(targets, preds, average="macro")
    except:
        roc, pr = 0, 0

    print("\nRESULTS")
    print("Train Loss:", train_loss)
    print("Val Loss:", val_loss / len(val_loader))
    print("ROC-AUC:", roc)
    print("PR-AUC:", pr)

print("\nDONE ✔")