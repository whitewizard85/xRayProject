import os
import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from tqdm import tqdm

from dataset import NIHChestXrayDataset, collate_fn

# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nDEVICE:", device)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode"

TEST_CSV = os.path.join(BASE_DIR, "test_split.csv")
IMAGE_ROOT = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "Advanced_DenseNet169_v1.pth"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "optimized_thresholds_DenseNet169_v1.json"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "error_analysis_DenseNet169_v1.csv"
)

# =========================================================
# CLASSES
# =========================================================

classes = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax",
    "Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia","No Finding"
]

num_classes = len(classes)

# =========================================================
# DATA
# =========================================================

test_df = pd.read_csv(TEST_CSV)

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

dataset = NIHChestXrayDataset(
    test_df,
    image_root=IMAGE_ROOT,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=1,
    collate_fn=collate_fn
)

print("\nTest samples:", len(dataset))

# =========================================================
# MODEL
# =========================================================

print("\nModel loading...")

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = models.densenet169(weights=None)
model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

model.load_state_dict(checkpoint["model"])
model = model.to(device)
model.eval()

print("Model loaded ✔")

# =========================================================
# THRESHOLDS
# =========================================================

with open(THRESHOLD_PATH, "r") as f:
    thresholds = json.load(f)

thresholds = np.array([thresholds[c] for c in classes])

print("Thresholds loaded ✔")

# =========================================================
# INFERENCE
# =========================================================

all_probs = []
all_targets = []
all_names = []

print("\nRunning inference...")

with torch.no_grad():
    for images, labels in tqdm(loader):

        if images is None:
            continue

        images = images.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)

        all_probs.append(probs.cpu())
        all_targets.append(labels)

all_probs = torch.cat(all_probs).numpy()
all_targets = torch.cat(all_targets).numpy()

# =========================================================
# ERROR ANALYSIS
# =========================================================

results = []

print("\n========================")
print("ERROR ANALYSIS REPORT")
print("========================")

for i, cls in enumerate(classes):

    y_true = all_targets[:, i]
    y_prob = all_probs[:, i]
    y_pred = (y_prob >= thresholds[i]).astype(int)

    fp = np.where((y_pred == 1) & (y_true == 0))[0]
    fn = np.where((y_pred == 0) & (y_true == 1))[0]

    print(f"\n{cls}")
    print(f"  False Positives: {len(fp)}")
    print(f"  False Negatives: {len(fn)}")

    for idx in fp[:3]:
        results.append([
            cls, "FP",
            test_df.iloc[idx]["Image Index"],
            float(y_prob[idx])
        ])

    for idx in fn[:3]:
        results.append([
            cls, "FN",
            test_df.iloc[idx]["Image Index"],
            float(y_prob[idx])
        ])

# =========================================================
# SAVE
# =========================================================

df = pd.DataFrame(
    results,
    columns=["Class", "Type", "Image", "Score"]
)

df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved ✔")
print(OUTPUT_CSV)