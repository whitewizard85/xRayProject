import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from tqdm import tqdm

from dataset import NIHChestXrayDataset, collate_fn

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nDEVICE:", device)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode"

TEST_CSV = os.path.join(BASE_DIR, "test_split.csv")

IMAGE_ROOT = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "Advanced_ResNet50_v1.pth"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "optimized_thresholds_ResNet50_v1.json"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "error_analysis_ResNet50_v1.csv"
)

# =====================================================
# CLASSES
# =====================================================

classes = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax",
    "Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia","No Finding"
]

num_classes = len(classes)

# =====================================================
# DATA
# =====================================================

IMAGE_SIZE = 320

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_df = pd.read_csv(TEST_CSV)

test_dataset = NIHChestXrayDataset(
    test_df,
    image_root=IMAGE_ROOT,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    collate_fn=collate_fn
)

print("\nTest samples:", len(test_dataset))

# =====================================================
# LOAD MODEL
# =====================================================

print("\nModel loading...")

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, num_classes)
)

model.load_state_dict(checkpoint["model"])
model = model.to(device)
model.eval()

print("Model loaded ✔")

# =====================================================
# LOAD THRESHOLDS
# =====================================================

with open(THRESHOLD_PATH, "r") as f:
    thresholds = json.load(f)

print("Thresholds loaded ✔")

# =====================================================
# INFERENCE
# =====================================================

print("\nRunning inference...")

all_preds = []
all_targets = []

with torch.no_grad():

    for images, labels in tqdm(test_loader):

        if images is None:
            continue

        images = images.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)

        all_preds.append(probs.cpu().numpy())
        all_targets.append(labels.numpy())

preds = np.concatenate(all_preds)
targets = np.concatenate(all_targets)

# =====================================================
# ERROR ANALYSIS
# =====================================================

print("\n========================")
print("ERROR ANALYSIS REPORT")
print("========================")

rows = []

for i, cls in enumerate(classes):

    thr = thresholds[cls]

    y_prob = preds[:, i]
    y_true = targets[:, i]

    y_pred = (y_prob >= thr).astype(int)

    fp_idx = np.where((y_pred == 1) & (y_true == 0))[0]
    fn_idx = np.where((y_pred == 0) & (y_true == 1))[0]

    fp_count = len(fp_idx)
    fn_count = len(fn_idx)

    print(f"\n{cls}")
    print(f"  False Positives: {fp_count}")
    print(f"  False Negatives: {fn_count}")

    # esempi
    fp_examples = [
        (test_df.iloc[i]["Image Index"], float(y_prob[i]))
        for i in fp_idx[:3]
    ]

    fn_examples = [
        (test_df.iloc[i]["Image Index"], float(y_prob[i]))
        for i in fn_idx[:3]
    ]

    rows.append({
        "class": cls,
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "fp_examples": fp_examples,
        "fn_examples": fn_examples
    })

# =====================================================
# SAVE CSV
# =====================================================

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved ✔")
print(OUTPUT_CSV)