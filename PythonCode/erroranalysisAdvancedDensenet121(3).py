import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torchvision import models, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import NIHChestXrayDataset

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDEVICE:", device)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "Advanced_DenseNet121_v2.pth"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "optimized_thresholds_v2.json"
)

TEST_CSV = os.path.join(
    BASE_DIR,
    "test_split.csv"
)

IMAGE_ROOT = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "error_analysis_DenseNet121_v2.csv"
)

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
# TRANSFORMS
# =====================================================

IMAGE_SIZE = 320

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =====================================================
# DATASET
# =====================================================

test_df = pd.read_csv(TEST_CSV)

dataset = NIHChestXrayDataset(
    test_df,
    image_root=IMAGE_ROOT,
    transform=test_transform
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=1,
    pin_memory=True
)

print("Test samples:", len(dataset))

# =====================================================
# MODEL
# =====================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = models.densenet121()

model.classifier = nn.Linear(
    model.classifier.in_features,
    num_classes
)

model.load_state_dict(
    checkpoint["model"]
)

model.to(device)
model.eval()

print("Model loaded ✔")

# =====================================================
# THRESHOLDS
# =====================================================

with open(THRESHOLD_PATH, "r") as f:
    thresholds_dict = json.load(f)

thresholds = np.array([
    thresholds_dict[c]
    for c in classes
])

print("Thresholds loaded ✔")

# =====================================================
# INFERENCE
# =====================================================

all_probs = []
all_targets = []

print("\nRunning inference...")

with torch.no_grad():

    for images, labels in tqdm(loader):

        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        all_probs.append(
            probs.cpu().numpy()
        )

        all_targets.append(
            labels.numpy()
        )

all_probs = np.vstack(all_probs)
all_targets = np.vstack(all_targets)

# =====================================================
# THRESHOLD
# =====================================================

preds = (
    all_probs >= thresholds
).astype(int)

# =====================================================
# ERROR ANALYSIS
# =====================================================

rows = []

print("\n========================")
print("ERROR ANALYSIS REPORT")
print("========================")

for c in range(num_classes):

    cls = classes[c]

    fp_mask = (
        (preds[:, c] == 1) &
        (all_targets[:, c] == 0)
    )

    fn_mask = (
        (preds[:, c] == 0) &
        (all_targets[:, c] == 1)
    )

    fp_count = int(fp_mask.sum())
    fn_count = int(fn_mask.sum())

    print(f"\n{cls}")
    print(f"  False Positives: {fp_count}")
    print(f"  False Negatives: {fn_count}")

    rows.append({
        "Class": cls,
        "False Positives": fp_count,
        "False Negatives": fn_count,
        "FP/FN Ratio":
            round(
                fp_count / (fn_count + 1e-6),
                3
            )
    })

# =====================================================
# SAVE CSV
# =====================================================

report_df = pd.DataFrame(rows)

report_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nSaved ✔")
print(OUTPUT_CSV)