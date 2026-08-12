import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.metrics import f1_score
from tqdm import tqdm

# =====================================================
# PATHS
# =====================================================

BASE_DIR = "/home/gpuvm/Desktop/Luca Migliaccio"
PYTHON_DIR = os.path.join(BASE_DIR, "PythonCode")

MODEL_PATH = os.path.join(
    PYTHON_DIR,
    "checkpoints",
    "Advanced_DenseNet121.pth"
)

VAL_CSV = os.path.join(PYTHON_DIR, "val_split.csv")

IMAGE_ROOT = os.path.join(BASE_DIR, "archive")

# =====================================================
# CLASSES
# =====================================================

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

# =====================================================
# ENCODE LABELS
# =====================================================

def encode_labels(label_str):

    vec = torch.zeros(num_classes, dtype=torch.float32)

    labels = label_str.split("|")

    if "No Finding" in labels:
        vec[-1] = 1.0
        return vec

    for l in labels:
        if l in classes:
            vec[classes.index(l)] = 1.0

    return vec

# =====================================================
# IMAGE PATH
# =====================================================

def get_image_path(img_name):

    for i in range(1, 13):

        path = os.path.join(
            IMAGE_ROOT,
            f"images_{i:03d}",
            "images",
            img_name
        )

        if os.path.exists(path):
            return path

    return None

# =====================================================
# TRANSFORMS
# =====================================================

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# DATASET
# =====================================================

class NIHChestXrayDataset(Dataset):

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

# =====================================================
# COLLATE FUNCTION
# =====================================================

def collate_fn(batch):

    batch = [b for b in batch if b[0] is not None]

    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])

    return images, labels

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n========================")
print("DEVICE")
print("========================")
print(device)

# =====================================================
# LOAD VALIDATION DATA
# =====================================================

val_df = pd.read_csv(VAL_CSV)

val_loader = DataLoader(
    NIHChestXrayDataset(val_df, transform=val_transform),
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn
)

print("\nValidation samples:", len(val_df))

# =====================================================
# MODEL
# =====================================================

model = models.densenet121(weights=None)

model.classifier = nn.Linear(
    model.classifier.in_features,
    num_classes
)

# =====================================================
# LOAD MODEL
# =====================================================

print("\n========================")
print("LOADING MODEL")
print("========================")
print(MODEL_PATH)

checkpoint = torch.load(MODEL_PATH, map_location=device)

model.load_state_dict(checkpoint["model"])

model = model.to(device)

print("Model loaded ✔")

# =====================================================
# GENERATE PREDICTIONS
# =====================================================

model.eval()

all_probs = []
all_targets = []

print("\n========================")
print("GENERATING PREDICTIONS")
print("========================")

with torch.no_grad():

    for images, labels in tqdm(val_loader):

        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        all_probs.append(probs.cpu())
        all_targets.append(labels)

# =====================================================
# CONCAT
# =====================================================

all_probs = torch.cat(all_probs).numpy()
all_targets = torch.cat(all_targets).numpy()

print("\nPredictions shape:", all_probs.shape)
print("Targets shape:", all_targets.shape)

# =====================================================
# THRESHOLD TUNING
# =====================================================

print("\n========================")
print("THRESHOLD SEARCH")
print("========================")

best_thresholds = {}

for i, cls in enumerate(classes):

    best_f1 = 0
    best_threshold = 0.5

    y_true = all_targets[:, i]
    y_prob = all_probs[:, i]

    for threshold in np.arange(0.05, 1.00, 0.05):

        y_pred = (y_prob >= threshold).astype(int)

        try:

            score = f1_score(y_true, y_pred)

            if score > best_f1:

                best_f1 = score
                best_threshold = float(threshold)

        except:
            pass

    best_thresholds[cls] = round(best_threshold, 2)

    print(
        f"{cls:25s} | "
        f"Threshold: {best_threshold:.2f} | "
        f"F1: {best_f1:.4f}"
    )

# =====================================================
# SAVE THRESHOLDS
# =====================================================

threshold_path = os.path.join(
    PYTHON_DIR,
    "optimized_thresholds.json"
)

with open(threshold_path, "w") as f:
    json.dump(best_thresholds, f, indent=4)

print("\n========================")
print("THRESHOLDS SAVED")
print("========================")
print(threshold_path)

print("\nDONE ✔")