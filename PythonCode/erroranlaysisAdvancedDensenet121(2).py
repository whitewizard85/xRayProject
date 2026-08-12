import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader

from dataset import NIHChestXrayDataset
from preprocessing import val_transform

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDEVICE:", device)

# =========================
# PATHS
# =========================
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
model_path = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/checkpoints/Advanced_DenseNet121.pth"
threshold_path = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/optimized_thresholds.json"
image_root = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

# =========================
# LOAD DATA
# =========================
test_df = pd.read_csv(test_csv)

test_loader = DataLoader(
    NIHChestXrayDataset(test_df, image_root=image_root, transform=val_transform),
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("Test batches:", len(test_loader))

# =========================
# CLASSES
# =========================
classes = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax",
    "Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia","No Finding"
]

num_classes = len(classes)

# =========================
# MODEL
# =========================
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model = model.to(device)
model.eval()

print("\nModel loaded ✔")

# =========================
# THRESHOLDS
# =========================
with open(threshold_path, "r") as f:
    thresholds = json.load(f)

# =========================
# STORAGE
# =========================
all_probs = []
all_targets = []

print("\nRunning inference...")

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)

        all_probs.append(probs.cpu())
        all_targets.append(labels.cpu())

all_probs = torch.cat(all_probs).numpy()
all_targets = torch.cat(all_targets).numpy()

# =========================
# APPLY THRESHOLDS
# =========================
preds = np.zeros_like(all_probs)

for i, c in enumerate(classes):
    t = thresholds.get(c, 0.5)
    preds[:, i] = (all_probs[:, i] >= t).astype(int)

# =========================
# ERROR ANALYSIS STORAGE
# =========================
fp_results = {c: [] for c in classes}
fn_results = {c: [] for c in classes}

# =========================
# ANALYSIS LOOP
# =========================
for i in range(len(all_probs)):

    for j, c in enumerate(classes):

        y_true = all_targets[i, j]
        y_pred = preds[i, j]
        prob = all_probs[i, j]

        img_id = test_df.iloc[i]["Image Index"]

        # FALSE POSITIVE
        if y_true == 0 and y_pred == 1:
            fp_results[c].append((img_id, prob))

        # FALSE NEGATIVE
        if y_true == 1 and y_pred == 0:
            fn_results[c].append((img_id, prob))

# =========================
# REPORT PRINT
# =========================
print("\n========================")
print("ERROR ANALYSIS REPORT")
print("========================")

for c in classes:

    fps = fp_results[c]
    fns = fn_results[c]

    print(f"\n{c}")
    print(f"  False Positives: {len(fps)}")
    print(f"  False Negatives: {len(fns)}")

    if len(fps) > 0:
        print("  Example FP:", fps[:3])

    if len(fns) > 0:
        print("  Example FN:", fns[:3])

# =========================
# SAVE CSV REPORT
# =========================
rows = []

for c in classes:

    for img, p in fp_results[c]:
        rows.append([img, c, "FP", p])

    for img, p in fn_results[c]:
        rows.append([img, c, "FN", p])

df_errors = pd.DataFrame(rows, columns=["image", "class", "error_type", "probability"])

save_path = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/error_analysis.csv"
df_errors.to_csv(save_path, index=False)

print("\nSaved error report ✔")
print(save_path)