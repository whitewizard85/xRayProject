import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm


# =========================
# 1. DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n========================")
print("DEVICE")
print("========================")
print(device)


# =========================
# 2. PATHS
# =========================

root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

train_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"


# =========================
# 3. NIH CLASSES
# =========================

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
    "Hernia"
]

num_classes = len(classes)


# =========================
# 4. LABEL ENCODING
# =========================

def encode_labels(label_str):

    vec = torch.zeros(num_classes)

    labels = label_str.split("|")

    for l in labels:
        if l in classes:
            vec[classes.index(l)] = 1.0

    return vec


# =========================
# 5. FIND IMAGE
# =========================

def get_image_path(img_name):

    for i in range(1, 13):

        folder = f"images_{i:03d}"

        path = os.path.join(
            root_dir,
            folder,
            "images",
            img_name
        )

        if os.path.exists(path):
            return path

    return None


# =========================
# 6. DATASET
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
# 7. TRANSFORMS
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
# 8. LOAD CSV
# =========================

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)

print("\nTrain size:", len(train_df))
print("Validation size:", len(val_df))


# =========================
# 9. DATASETS
# =========================

train_dataset = NIHChestDataset(train_df, train_transform)
val_dataset = NIHChestDataset(val_df, val_transform)


# =========================
# 10. DATALOADERS
# =========================

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
# 11. MODEL
# =========================

model = models.densenet121(
    weights=models.DenseNet121_Weights.IMAGENET1K_V1
)

in_features = model.classifier.in_features

model.classifier = nn.Linear(in_features, num_classes)

model = model.to(device)

print("\nDenseNet121 loaded ✔")


# =========================
# 12. LOSS + OPTIMIZER
# =========================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)


# =========================
# 13. TRAINING LOOP
# =========================

epochs = 5

best_val_loss = float("inf")

print("\n========================")
print("START TRAINING")
print("========================")

for epoch in range(epochs):

    # =====================
    # TRAIN
    # =====================

    model.train()

    train_loss = 0.0

    loop = tqdm(train_loader)

    for images, labels in loop:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)

    # =====================
    # VALIDATION
    # =====================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # =====================
    # SAVE BEST MODEL
    # =====================

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        torch.save(
            model.state_dict(),
            "best_densenet121.pth"
        )

        print("Best model saved ✔")


print("\n========================")
print("TRAINING COMPLETED")
print("========================")