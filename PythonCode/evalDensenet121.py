import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    roc_auc_score,
    classification_report
)

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

test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"

model_path = "/home/gpuvm/Desktop/Luca Migliaccio/best_densenet121.pth"


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

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# =========================
# 8. LOAD TEST CSV
# =========================

test_df = pd.read_csv(test_csv)

print("\nTest samples:", len(test_df))


# =========================
# 9. TEST DATASET
# =========================

test_dataset = NIHChestDataset(
    test_df,
    test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=1,
    pin_memory=True
)


# =========================
# 10. LOAD MODEL
# =========================

model = models.densenet121(
    weights=None
)

in_features = model.classifier.in_features

model.classifier = nn.Linear(
    in_features,
    num_classes
)

model.load_state_dict(
    torch.load(model_path)
)

model = model.to(device)

model.eval()

print("\nModel loaded ✔")


# =========================
# 11. EVALUATION
# =========================

all_labels = []
all_probs = []

print("\n========================")
print("RUNNING EVALUATION")
print("========================")

with torch.no_grad():

    loop = tqdm(test_loader)

    for images, labels in loop:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())


# =========================
# 12. CONCAT RESULTS
# =========================

all_labels = np.concatenate(all_labels, axis=0)
all_probs = np.concatenate(all_probs, axis=0)


# =========================
# 13. ROC-AUC
# =========================

print("\n========================")
print("ROC-AUC PER CLASS")
print("========================")

auc_scores = []

for i, cls in enumerate(classes):

    try:

        auc = roc_auc_score(
            all_labels[:, i],
            all_probs[:, i]
        )

        auc_scores.append(auc)

        print(f"{cls}: {auc:.4f}")

    except:

        print(f"{cls}: AUC ERROR")


# =========================
# 14. MEAN AUC
# =========================

mean_auc = np.mean(auc_scores)

print("\n========================")
print("MEAN ROC-AUC")
print("========================")

print(f"Mean AUC: {mean_auc:.4f}")


# =========================
# 15. THRESHOLD PREDICTIONS
# =========================

preds = (all_probs >= 0.5).astype(int)


# =========================
# 16. CLASSIFICATION REPORT
# =========================

print("\n========================")
print("CLASSIFICATION REPORT")
print("========================")

report = classification_report(
    all_labels,
    preds,
    target_names=classes,
    zero_division=0
)

print(report)


# =========================
# 17. FINAL MESSAGE
# =========================

print("\n========================")
print("EVALUATION COMPLETED")
print("========================")