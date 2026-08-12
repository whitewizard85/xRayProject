import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score
import torchxrayvision as xrv

# =====================================================
# CONFIGURAZIONE PATHS INTERRELAZIONATI
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"

# Checkpoints Modelli
model_path_v4 = "checkpoints/best_densenet121_v4_xrv.pth"
model_path_v5 = "checkpoints/best_resnet50_v5_xrv.pth"

# JSON Soglie Ottimizzate
thresholds_path_v4 = "checkpoints/optimized_thresholds_v4_xrv.json"
thresholds_path_v5 = "checkpoints/optimized_thresholds_v5_xrv.json"

# =====================================================
# COSTANTI
# =====================================================
IMAGE_SIZE = 512

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)

# =====================================================
# UTILS & DATASET (Standardizzato XRV)
# =====================================================
def encode_labels(label_str):
    vec = torch.zeros(num_classes)
    labels = str(label_str).split("|")
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

class NIHChestDatasetXRV(Dataset):
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

        image = Image.open(img_path).convert("L")
        
        img_np = np.array(image)
        img_np = xrv.datasets.normalize(img_np, maxval=255)
        image = Image.fromarray(img_np)

        label = encode_labels(label_str)

        if self.transform:
            image = self.transform(image)

        return image, label

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0: return torch.empty(0), torch.empty(0)
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return images, labels

# =====================================================
# CARICAMENTO FILTRI SOGLIE E COSTRUZIONE ENSEMBLE T
# =====================================================
print(f"Caricamento soglie V4 da: {thresholds_path_v4}")
with open(thresholds_path_v4, "r") as f:
    t_v4 = json.load(f)

print(f"Caricamento soglie V5 da: {thresholds_path_v5}")
with open(thresholds_path_v5, "r") as f:
    t_v5 = json.load(f)

# Creazione della soglia fusa (Media Matematica)
ensemble_thresholds = {}
print("\n--- SOGLIE FUSE PER L'ENSEMBLE ---")
for c in classes:
    ensemble_thresholds[c] = float((t_v4[c] + t_v5[c]) / 2.0)
    print(f"  -> {c:<20} | T-v4: {t_v4[c]:.2f} | T-v5: {t_v5[c]:.2f} -> T-Ensemble: {ensemble_thresholds[c]:.2f}")

# =====================================================
# PREPARAZIONE PIPELINE DATI (Risoluzione a 512px)
# =====================================================
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_df = pd.read_csv(test_csv)
test_loader = DataLoader(
    NIHChestDatasetXRV(test_df, test_transform),
    batch_size=16, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn
)

# =====================================================
# INIZIALIZZAZIONE ARCHITETTURE CUSTOM E CARICAMENTO PESI
# =====================================================
# 1. Modello V4 (DenseNet)
base_densenet = xrv.models.DenseNet(weights="densenet121-res224-all")
class XRVFeatureExtractor(nn.Module):
    def __init__(self, xrv_model, num_classes):
        super(XRVFeatureExtractor, self).__init__()
        self.features = xrv_model.features           
        self.classifier = nn.Linear(1024, num_classes) 

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

model_v4 = XRVFeatureExtractor(base_densenet, num_classes)
model_v4.load_state_dict(torch.load(model_path_v4, map_location=device))
model_v4 = model_v4.to(device).eval()
print(f"\n[OK] DenseNet V4 caricata correttamente.")

# 2. Modello V5 (ResNet50)
base_resnet = xrv.models.ResNet(weights="resnet50-res512-all")
class XRVResNetFeatureExtractor(nn.Module):
    def __init__(self, xrv_resnet, num_classes):
        super(XRVResNetFeatureExtractor, self).__init__()
        self.base_resnet = xrv_resnet
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.base_resnet.features(x)
        out = self.classifier(features)
        return out

model_v5 = XRVResNetFeatureExtractor(base_resnet, num_classes)
model_v5.load_state_dict(torch.load(model_path_v5, map_location=device))
model_v5 = model_v5.to(device).eval()
print(f"[OK] ResNet50 V5 caricata correttamente.")

print("\nConfigurazione Ensemble Pronta. Avvio inferenza parallela sul Test Set...")

# =====================================================
# LOOP DI INFERENZA CONGIUNTA (LATE FUSION)
# =====================================================
ensemble_probs = []
all_targets = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Ensemble Inference"):
        if images.numel() == 0: continue
        images = images.to(device)
        
        with torch.cuda.amp.autocast():
            outputs_v4 = model_v4(images)
            outputs_v5 = model_v5(images)
            
        probs_v4 = torch.sigmoid(outputs_v4).cpu().numpy()
        probs_v5 = torch.sigmoid(outputs_v5).cpu().numpy()
        
        # Late Fusion: Media delle probabilità predette
        batch_ensemble_probs = (probs_v4 + probs_v5) / 2.0
        
        ensemble_probs.append(batch_ensemble_probs)
        all_targets.append(labels.cpu().numpy())

ensemble_probs = np.vstack(ensemble_probs)
all_targets = np.vstack(all_targets)

# =====================================================
# CALCOLO METRICHE FINALI ENSEMBLE
# =====================================================
print("\n" + "="*60)
print("FINAL CLASSIFICATION REPORT (TEST SET - ENSEMBLE V4 + V5)")
print("="*60)

preds_bin = np.zeros_like(ensemble_probs)
for j, c in enumerate(classes):
    t = ensemble_thresholds[c]
    preds_bin[:, j] = (ensemble_probs[:, j] >= t).astype(int)

# Sanitizzazione tipi
all_targets = all_targets.astype(np.int32)
preds_bin = preds_bin.astype(np.int32)
ensemble_probs = ensemble_probs.astype(np.float32)

# Generazione report di classificazione binaria
report = classification_report(all_targets, preds_bin, target_names=classes, zero_division=0)
print(report)

print("\nROC-AUC PURE SUL TEST SET (ENSEMBLE V4 + V5):")
auc_dict = {}
auc_list = []
for j, c in enumerate(classes):
    try:
        auc_c = roc_auc_score(all_targets[:, j], ensemble_probs[:, j])
        auc_list.append(auc_c)
        auc_dict[c] = float(auc_c)
        print(f"  -> {c:<20}: {auc_c:.4f}")
    except ValueError:
        print(f"  -> {c:<20}: AUC non calcolabile")

macro_auc = np.mean(auc_list)
print(f"\n➔ MACRO AVERAGE ROC-AUC FINALE DELL'ENSEMBLE: {macro_auc:.4f}")

# Salvataggio dati strutturati per la tesi
output_results = {
    "macro_auc": float(macro_auc),
    "auc_per_class": auc_dict,
    "applied_thresholds": ensemble_thresholds
}
os.makedirs("checkpoints", exist_ok=True)
with open("checkpoints/final_ensemble_metrics.json", "w") as f:
    json.dump(output_results, f, indent=4)
print("\nRisultati dell'Ensemble salvati in checkpoints/final_ensemble_metrics.json ✔")