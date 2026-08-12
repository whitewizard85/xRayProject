import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torchvision import models, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import NIHChestXrayDataset, collate_fn

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n========================")
print("DEVICE")
print("========================")
print(device)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode"

MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "EfficientNetB3_ASL.pth")
THRESHOLD_PATH = os.path.join(BASE_DIR, "optimized_thresholds_EfficientNetB3.json")
TEST_CSV = os.path.join(BASE_DIR, "test_split.csv")
IMAGE_ROOT = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

OUT_PATH = os.path.join(BASE_DIR, "error_analysis_EfficientNetB3.json")

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
# TRANSFORM
# =====================================================

IMAGE_SIZE = 300

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# DATA
# =====================================================

df = pd.read_csv(TEST_CSV)

dataset = NIHChestXrayDataset(df, IMAGE_ROOT, transform)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    collate_fn=collate_fn
)

print("\nTest samples:", len(dataset))

# =====================================================
# MODEL
# =====================================================

print("\nLoading model...")

ckpt = torch.load(MODEL_PATH, map_location=device)

model = models.efficientnet_b3(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()

# =====================================================
# THRESHOLDS
# =====================================================

with open(THRESHOLD_PATH, "r") as f:
    thresholds_dict = json.load(f)

thresholds = np.array([thresholds_dict[c] for c in classes])

# =====================================================
# INFERENCE
# =====================================================

all_probs = []
all_targets = []

print("\nRunning inference...")

with torch.no_grad():
    for images, labels in tqdm(loader):

        if images is None:
            continue

        images = images.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)

        all_probs.append(probs.cpu().numpy())
        all_targets.append(labels.numpy())

all_probs = np.vstack(all_probs)
all_targets = np.vstack(all_targets)

# =====================================================
# BINARY PREDICTIONS
# =====================================================

preds = (all_probs >= thresholds).astype(int)

# =====================================================
# ERROR ANALYSIS
# =====================================================

analysis = {}

for i, cls in enumerate(classes):

    y_true = all_targets[:, i]
    y_prob = all_probs[:, i]
    y_pred = preds[:, i]

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    # safe stats
    pos_rate = np.mean(y_true) if len(y_true) > 0 else 0
    pred_pos_rate = np.mean(y_pred) if len(y_pred) > 0 else 0

    true_pos_conf = y_prob[y_true == 1]
    false_pos_conf = y_prob[(y_true == 0) & (y_pred == 1)]
    false_neg_conf = y_prob[(y_true == 1) & (y_pred == 0)]

    analysis[cls] = {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "positive_rate": float(pos_rate),
        "pred_positive_rate": float(pred_pos_rate),
        "avg_conf_true_pos": float(np.mean(true_pos_conf)) if len(true_pos_conf) > 0 else 0,
        "avg_conf_false_pos": float(np.mean(false_pos_conf)) if len(false_pos_conf) > 0 else 0,
        "avg_conf_false_neg": float(np.mean(false_neg_conf)) if len(false_neg_conf) > 0 else 0,
    }

    print(f"{cls:20s} | FP:{fp:5d} FN:{fn:5d} TP:{tp:5d}")

# =====================================================
# GLOBAL ERROR RANKING (IMAGE LEVEL)
# =====================================================

fp_scores = np.sum((preds == 1) & (all_targets == 0), axis=1)
fn_scores = np.sum((preds == 0) & (all_targets == 1), axis=1)

worst_fp_idx = np.argsort(-fp_scores)[:20]
worst_fn_idx = np.argsort(-fn_scores)[:20]

analysis["global"] = {
    "top_false_positive_images": worst_fp_idx.tolist(),
    "top_false_negative_images": worst_fn_idx.tolist()
}

# =====================================================
# SAVE
# =====================================================

with open(OUT_PATH, "w") as f:
    json.dump(analysis, f, indent=4)

print("\nSaved ✔")
print(OUT_PATH)