import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    f1_score
)

# =====================================================
# PATHS
# =====================================================
PYTHON_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode"
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

TEST_CSV = os.path.join(PYTHON_DIR, "test_split.csv")
MODEL_PATH = "best_densenet121_v2.pth"  # I pesi salvati dal train
THRESHOLD_PATH = os.path.join(PYTHON_DIR, "optimized_thresholds.json") # Le soglie appena calcolate

# =====================================================
# CLASSES (Le 14 classi reali del tuo modello)
# =====================================================
classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)

# =====================================================
# LABEL ENCODING
# =====================================================
def encode_labels(label_str):
    vec = torch.zeros(num_classes)
    labels = label_str.split("|")
    for l in labels:
        if l in classes:
            vec[classes.index(l)] = 1.0
    return vec

# =====================================================
# IMAGE PATH
# =====================================================
def get_image_path(img_name):
    for i in range(1, 13):
        folder = f"images_{i:03d}"
        path = os.path.join(root_dir, folder, "images", img_name)
        if os.path.exists(path):
            return path
    return None

# =====================================================
# DATASET
# =====================================================
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
        if img_path is None:
            return None, None

        image = Image.open(img_path).convert("RGB")
        label = encode_labels(label_str)

        if self.transform:
            image = self.transform(image)

        return image, label

# =====================================================
# COLLATE FUNCTION (Per saltare i file mancanti in sicurezza)
# =====================================================
def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0)
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return images, labels

# =====================================================
# IMAGE SIZE & TRANSFORMS (Allineato a 384 del Train)
# =====================================================
IMAGE_SIZE = 384

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDEVICE:", device)

# =====================================================
# LOAD TEST DATA
# =====================================================
if not os.path.exists(TEST_CSV):
    raise FileNotFoundError(f"Non ho trovato il file di test in: {TEST_CSV}")

test_df = pd.read_csv(TEST_CSV)
test_dataset = NIHChestDataset(test_df, test_transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn
)

print(f"\nSamples di Test rilevati nel CSV: {len(test_df)}")

# =====================================================
# LOAD MODEL
# =====================================================
print("\nInizializzazione architettura modello...")
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

print(f"Caricamento pesi da: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"File dei pesi non trovato: {MODEL_PATH}")

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print("Modello caricato e impostato in eval mode ✔")

# =====================================================
# LOAD THRESHOLDS
# =====================================================
print(f"Caricamento soglie ottimizzate da: {THRESHOLD_PATH}")
if not os.path.exists(THRESHOLD_PATH):
    raise FileNotFoundError(f"File delle soglie JSON non trovato: {THRESHOLD_PATH}")

with open(THRESHOLD_PATH, "r") as f:
    thresholds_dict = json.load(f)

# Genera un array numpy ordinato esattamente come la lista delle classi
thresholds = np.array([thresholds_dict[c] for c in classes])
print("Soglie caricate con successo ✔")

# =====================================================
# RUN INFERENCE ON TEST SET
# =====================================================
all_probs = []
all_targets = []

print("\n========================")
# Avvio inferenza
print("AVVIO INFERENZA SUL TEST SET")
print("========================")

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        if images.numel() == 0:
            continue

        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)

        all_probs.append(probs.cpu().numpy())
        all_targets.append(labels.numpy())

all_probs = np.vstack(all_probs)
all_targets = np.vstack(all_targets)

print(f"\nPredizioni completate. Matrix shape: {all_probs.shape}")

# =====================================================
# METRICS GENERATION
# =====================================================
print("\n========================")
print("ROC-AUC PER CLASS")
print("========================")

roc_scores = []
for i, cls in enumerate(classes):
    try:
        auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
        roc_scores.append(auc)
        print(f"{cls:25s} | ROC-AUC: {auc:.4f}")
    except ValueError:
        # Gestione di sicurezza nel caso una classe rara non sia presente nel test split
        print(f"{cls:25s} | ROC-AUC: N/A (Nessun sample positivo nel test set)")

mean_roc = np.mean(roc_scores)

# Calcolo Precision-Recall AUC (Macro)
pr_auc = average_precision_score(all_targets, all_probs, average="macro")

# Applicazione delle soglie ottimizzate per la binarizzazione
pred_binary = (all_probs >= thresholds).astype(int)

# Generazione del report testuale completo per classe
report = classification_report(
    all_targets,
    pred_binary,
    target_names=classes,
    digits=4,
    zero_division=0
)

# Calcolo dei vari F1-Score globali
macro_f1 = f1_score(all_targets, pred_binary, average="macro", zero_division=0)
micro_f1 = f1_score(all_targets, pred_binary, average="micro", zero_division=0)
samples_f1 = f1_score(all_targets, pred_binary, average="samples", zero_division=0)

# =====================================================
# SHOW RESULTS
# =====================================================
print("\n========================")
print("GLOBAL TEST METRICS")
print("========================")
print(f"Mean ROC-AUC : {mean_roc:.4f}")
print(f"PR-AUC       : {pr_auc:.4f}")
print(f"Macro F1     : {macro_f1:.4f}")
print(f"Micro F1     : {micro_f1:.4f}")
print(f"Samples F1   : {samples_f1:.4f}")
print("\n" + report)

# =====================================================
# SAVE COMPREHENSIVE REPORT
# =====================================================
report_path = os.path.join(PYTHON_DIR, "evaluation_DenseNet121_v2.txt")

with open(report_path, "w") as f:
    f.write("=========================================\n")
    f.write("DenseNet121 v2 (384x384) - TEST EVALUATION\n")
    f.write("=========================================\n\n")
    f.write(f"Mean ROC-AUC : {mean_roc:.4f}\n")
    f.write(f"PR-AUC       : {pr_auc:.4f}\n")
    f.write(f"Macro F1     : {macro_f1:.4f}\n")
    f.write(f"Micro F1     : {micro_f1:.4f}\n")
    f.write(f"Samples F1   : {samples_f1:.4f}\n\n")
    f.write("CLASSIFICATION REPORT PER CLASS:\n")
    f.write("-----------------------------------------\n")
    f.write(report)

print("\n========================")
print("REPORT SALVATO CON SUCCESSO ✔")
print("========================")
print(report_path)