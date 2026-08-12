import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm


# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDEVICE:", device)


# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

train_csv = os.path.join(BASE_DIR, "train_split.csv")
val_csv = os.path.join(BASE_DIR, "val_split.csv")


# =========================
# CLASSES
# =========================

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
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
# IMAGE PATH SEARCH
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

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = encode_labels(label_str)

        return image, label


# =========================
# TRANSFORMS
# =========================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# =========================
# DATA
# =========================

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)

print("\nTrain size:", len(train_df))
print("Val size:", len(val_df))


train_dataset = NIHChestDataset(train_df, train_transform)
val_dataset = NIHChestDataset(val_df, val_transform)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=1,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=1,
    pin_memory=True
)


# =========================
# MODEL
# =========================

model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, num_classes)

model = model.to(device)

print("\nDenseNet121 loaded ✔")


# =========================
# LOSS + OPTIMIZER
# =========================

pos_weight = torch.tensor([
    1.2, 2.5, 1.1, 0.8,
    2.0, 2.2, 3.0, 1.8,
    1.4, 2.8, 2.0, 2.5,
    1.6, 4.0
]).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


# =========================
# AMP (GPU SPEEDUP)
# =========================

scaler = torch.cuda.amp.GradScaler()


# =========================
# TRAIN SETTINGS
# =========================

epochs = 20
best_val_loss = float("inf")
patience = 5
counter = 0


# =========================
# TRAIN LOOP
# =========================

print("\nSTART TRAINING")

for epoch in range(epochs):

    # TRAIN
    model.train()
    train_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

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


    # VALIDATION
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

    val_loss /= len(val_loader)


    print("\n========================")
    print(f"Epoch {epoch+1}")
    print("========================")
    print("Train Loss:", round(train_loss, 4))
    print("Val Loss:", round(val_loss, 4))


    # SAVE BEST
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        counter = 0

        torch.save(model.state_dict(), "best_densenet121.pth")

        print("Best model saved ✔")

    else:
        counter += 1

    # EARLY STOP
    if counter >= patience:
        print("\nEarly stopping triggered ⛔")
        break


print("\nTRAINING COMPLETED ✔")