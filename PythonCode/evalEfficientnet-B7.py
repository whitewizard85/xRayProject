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
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve, auc

# =====================================================
# 1. CONFIGURAZIONE PERCORSI E DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDEVICE IN USO:", device)

BASE_DIR = "/home/gpuvm/Desktop/Luca Migliaccio"
PYTHON_DIR = os.path.join(BASE_DIR, "PythonCode")
root_dir = os.path.join(BASE_DIR, "archive")

TEST_CSV = os.path.join(PYTHON_DIR, "test_split.csv")
MODEL_PATH = os.path.join(BASE_DIR, "best_efficientnet_b7.pth")
JSON_THRESHOLDS_PATH = os.path.join(PYTHON_DIR, "optimized_thresholds_EfficientNetB7.json")

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
NUM_CLASSES = len(classes)
BATCH_SIZE = 16  
IMAGE_SIZE = 600

# =====================================================
# 2. DEFINIZIONE DATASET (Allineato al train 600x600)
# =====================================================
def encode_labels(label_str):
    vec = torch.zeros(NUM_CLASSES)
    labels = label_str.split("|")
    for l in labels:
        if l in classes:
            vec[classes.index(l)] = 1.0
    return vec

def get_image_path(img_name):
    for i in range(1, 13):
        folder = f"images_{i:03d}"
        path = os.path.join(root_dir, folder, "images", img_name)
        if os.path.exists(path):
            return path
    return None

class NIHChestTestDataset(Dataset):
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

# =====================================================
# 3. TRASFORMAZIONI E DATALOADER
# =====================================================
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize
])

test_df = pd.read_csv(TEST_CSV)
test_dataset = NIHChestTestDataset(test_df, transform=test_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,  
    pin_memory=True
)

print("Campioni inseriti nel Test Set:", len(test_dataset))

# =====================================================
# 4. ARCHITETTURA EFFICIENTNET-B7 E CARICAMENTO PESI
# =====================================================
print("\nInizializzazione EfficientNet-B7...")
model = models.efficientnet_b7(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"File pesi non trovato in: {MODEL_PATH}")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("Modello EfficientNet-B7 caricato correttamente ✔")

# =====================================================
# 5. CARICAMENTO SOGLIE OTTENUTE DA JSON
# =====================================================
if not os.path.exists(JSON_THRESHOLDS_PATH):
    raise FileNotFoundError(f"File delle soglie ottimizzate non trovato in: {JSON_THRESHOLDS_PATH}")

with open(JSON_THRESHOLDS_PATH, "r") as f:
    thresholds_dict = json.load(f)
print("Soglie personalizzate caricate con successo ✔")

# =====================================================
# 6. INFERENZA (TEST SET)
# =====================================================
y_true = []
y_pred = []

print("\nEsecuzione inferenza sul Test Set...")
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            
        probs = torch.sigmoid(outputs)
        
        y_true.append(labels.cpu().numpy())
        y_pred.append(probs.cpu().numpy())

y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)

# =====================================================
# 7. CALCOLO ROC-AUC PER CLASS
# =====================================================
print("\n========================")
print("ROC-AUC PER CLASS")
print("========================")

auc_scores = []
for i, cls in enumerate(classes):
    try:
        auc_val = roc_auc_score(y_true[:, i], y_pred[:, i])
        auc_scores.append(auc_val)
        print(f"{cls:25s}: {auc_val:.4f}")
    except Exception as e:
        print(f"{cls:25s}: NON CALCOLABILE")

print("\n========================")
print("MEAN ROC-AUC")
print("========================")
print("Mean AUC finale sul Test Set:", round(np.mean(auc_scores), 4))

# =====================================================
# 7.1 CALCOLO MICRO ROC-AUC E MACRO PR-AUC
# =====================================================
micro_auc = roc_auc_score(y_true, y_pred, average="micro")
pr_auc_list = []
for i in range(NUM_CLASSES):
    p_c, r_c, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
    pr_auc_list.append(auc(r_c, p_c))
macro_pr_auc = np.mean(pr_auc_list)

print("\n========================")
print("METRICHE AGGIUNTIVE")
print("========================")
print(f"Micro ROC-AUC : {micro_auc:.4f}")
print(f"Macro PR-AUC  : {macro_pr_auc:.4f}")

# =====================================================
# 8. APPLICAZIONE SOGLIE E FIX DEL CRASH TIPO DATO
# =====================================================
y_pred_bin = np.zeros_like(y_pred)

for i, cls in enumerate(classes):
    class_thresh = thresholds_dict[cls]
    y_pred_bin[:, i] = (y_pred[:, i] >= class_thresh).astype(int)

# CONVERSIONE ESPLICITA IN INT32 PER SCIPY/SKLEARN
y_true_clean = y_true.astype(np.int32)
y_pred_bin_clean = y_pred_bin.astype(np.int32)

print("\n========================")
print("CLASSIFICATION REPORT (Soglie Ottimizzate)")
print("========================")
print(classification_report(y_true_clean, y_pred_bin_clean, target_names=classes, zero_division=0))