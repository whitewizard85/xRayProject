import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# =========================
# PATHS
# =========================

train_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv"
val_csv   = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
test_csv  = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"

image_root = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

# =========================
# LOAD SPLITS
# =========================

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

# =========================
# LABELS (14 + NO FINDING = 15)
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
    "Hernia",
    "No Finding"
]

num_classes = len(classes)


def encode_labels(label_str):

    vec = torch.zeros(num_classes, dtype=torch.float32)
    labels = label_str.split("|")

    # caso sano
    if "No Finding" in labels:
        vec[-1] = 1.0
        return vec

    for l in labels:
        if l in classes:
            vec[classes.index(l)] = 1.0

    return vec


# =========================
# IMAGE PATH RESOLUTION
# =========================

def get_image_path(img_name):
    for i in range(1, 13):
        path = os.path.join(image_root, f"images_{i:03d}", "images", img_name)
        if os.path.exists(path):
            return path
    return None


# =========================
# TRANSFORMS
# =========================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(7),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# =========================
# DATASET
# =========================

class NIHDataset(Dataset):

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
# COLLATE FUNCTION
# =========================

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]

    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])

    return images, labels


# =========================
# DATALOADERS
# =========================

train_loader = DataLoader(
    NIHDataset(train_df, train_transform),
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    NIHDataset(val_df, val_transform),
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    NIHDataset(test_df, val_transform),
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn
)

# =========================
# GPU CHECK
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n========================")
print("DATALOADER READY")
print("========================")
print("Device:", device)

print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))
print("Test batches:", len(test_loader))


# =========================
# SANITY CHECK
# =========================

if __name__ == "__main__":

    images, labels = next(iter(train_loader))

    print("\nBatch image shape:", images.shape)
    print("Batch label shape:", labels.shape)