import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDEVICE:", device)


# =========================
# PATHS
# =========================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

train_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv"
val_csv   = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"


# =========================
# CLASSES (NO NO FINDING)
# =========================
classes = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax",
    "Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia"
]

num_classes = len(classes)


# =========================
# LABEL ENCODING
# =========================
def encode_labels(label_str):
    vec = torch.zeros(num_classes)

    labels = label_str.split("|")
    for l in labels:
        if l in classes:
            vec[classes.index(l)] = 1.0

    return vec


# =========================
# IMAGE PATH
# =========================
def get_image_path(img_name):
    for i in range(1, 13):
        folder = f"images_{i:03d}"
        path = os.path.join(root_dir, folder, "images", img_name)

        if os.path.exists(path):
            return path
    return None


# =========================
# DATASET
# =========================
class NIHChestDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_name = row["Image Index"]
        label_str = row["Finding Labels"]

        img_path = get_image_path(img_name)
        if img_path is None:
            return None, None

        image = Image.open(img_path).convert("RGB")
        label = encode_labels(label_str)

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================
# IMAGE SIZE
# =========================
IMAGE_SIZE = 384


# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
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
# LOAD DATA
# =========================
train_df = pd.read_csv(train_csv)
val_df   = pd.read_csv(val_csv)

train_dataset = NIHChestDataset(train_df, train_transform)
val_dataset   = NIHChestDataset(val_df, val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# =========================
# MODEL
# =========================
model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model = model.to(device)

print("\nDenseNet121 loaded ✔")


# =========================
# LOSS
# =========================
criterion = nn.BCEWithLogitsLoss()


# =========================
# OPTIMIZER + SCHEDULER
# =========================
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

epochs = 30
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    steps_per_epoch=len(train_loader),
    epochs=epochs
)


# =========================
# MIXED PRECISION
# =========================
scaler = torch.cuda.amp.GradScaler()


# =========================
# EARLY STOPPING
# =========================
patience = 10
no_improve = 0
best_auc = 0


# =========================
# TRAINING
# =========================
print("\nSTART TRAINING")

for epoch in range(epochs):

    # ----------------
    # TRAIN
    # ----------------
    model.train()
    train_loss = 0

    for images, labels in tqdm(train_loader):

        if images is None:
            continue

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # ----------------
    # VALIDATION
    # ----------------
    model.eval()

    val_loss = 0
    preds = []
    targets = []

    with torch.no_grad():

        for images, labels in val_loader:

            if images is None:
                continue

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds.append(torch.sigmoid(outputs).cpu().numpy())
            targets.append(labels.cpu().numpy())

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    auc = roc_auc_score(targets, preds, average="macro")

    print(f"\nEpoch {epoch+1}/{epochs}")
    print("Train Loss:", round(train_loss, 4))
    print("Val Loss:", round(val_loss/len(val_loader), 4))
    print("ROC-AUC:", round(auc, 4))


    # ----------------
    # SAVE BEST
    # ----------------
    if auc > best_auc:
        best_auc = auc
        no_improve = 0

        torch.save(model.state_dict(), "best_densenet121_v2.pth")
        print("✔ BEST MODEL SAVED")

    else:
        no_improve += 1
        print(f"No improvement: {no_improve}/{patience}")

    # ----------------
    # EARLY STOPPING
    # ----------------
    if no_improve >= patience:
        print("\n⛔ EARLY STOPPING TRIGGERED")
        break


print("\nTRAINING COMPLETED")