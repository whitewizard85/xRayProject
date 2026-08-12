import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, 
    classification_report, 
    roc_curve, 
    average_precision_score, 
    precision_recall_fscore_support
)
import torchxrayvision as xrv

# CONFIGURAZIONE PATHS
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
model_path_v4 = "checkpoints/best_densenet121_v4_xrv.pth"
model_path_v5 = "checkpoints/best_resnet50_v5_xrv.pth"
saved_ensemble_path = "checkpoints/best_feature_fusion_ensemble.pth"

IMAGE_SIZE = 512
BATCH_SIZE = 16

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)

# DATASET & PIPELINE
def encode_labels(label_str):
    vec = torch.zeros(num_classes)
    labels = str(label_str).split("|")
    for l in labels:
        if l in classes: vec[classes.index(l)] = 1.0
    return vec

def get_image_path(img_name):
    for i in range(1, 13):
        folder = f"images_{i:03d}"
        path = os.path.join(root_dir, folder, "images", img_name)
        if os.path.exists(path): return path
    return None

class NIHChestDatasetXRV(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = get_image_path(img_name)
        if img_path is None: return None, None
        try:
            image = Image.open(img_path).convert("L")
            img_np = np.array(image)
            img_np = xrv.datasets.normalize(img_np, maxval=255)
            image = Image.fromarray(img_np)
            label = encode_labels(row["Finding Labels"])
            if self.transform: image = self.transform(image)
            return image, label
        except Exception: return None, None

def collate_fn(batch):
    batch = [b for b in batch if b is not None and b[0] is not None]
    if len(batch) == 0: return torch.empty(0), torch.empty(0)
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])

# ARCHITETTURA CORRETTA (Senza i link diretti a .base_resnet nel costruttore)
class FeatureFusionEnsemble(nn.Module):
    def __init__(self, path_densenet, path_resnet, num_classes):
        super(FeatureFusionEnsemble, self).__init__()
        self.dn_model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.rn_model = xrv.models.ResNet(weights="resnet50-res512-all")
        
        self.fusion_classifier = nn.Sequential(
            nn.Linear(1024 + 2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        f_dn = self.dn_model.features(x)
        if len(f_dn.shape) > 2:
            f_dn = F.adaptive_avg_pool2d(f_dn, (1, 1))
            f_dn = torch.flatten(f_dn, 1)
        
        f_rn = self.rn_model.features(x)
        if len(f_rn.shape) > 2:
            f_rn = F.adaptive_avg_pool2d(f_rn, (1, 1))
            f_rn = torch.flatten(f_rn, 1)
            
        f_combined = torch.cat((f_dn, f_rn), dim=1)
        return self.fusion_classifier(f_combined)

# INIZIALIZZAZIONE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeatureFusionEnsemble(model_path_v4, model_path_v5, num_classes).to(device)

# Carichiamo i pesi dell'addestramento appena concluso
model.load_state_dict(torch.load(saved_ensemble_path, map_location=device))
model.eval()

val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)
test_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

val_loader = DataLoader(NIHChestDatasetXRV(val_df, test_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)
test_loader = DataLoader(NIHChestDatasetXRV(test_df, test_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

# 1. PASSO DI VALIDAZIONE PER LE SOGLIE (Youden)
print("Estrazione probabilità sul Validation Set per calcolo soglie... 🔍")
val_preds, val_targets = [], []
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Validation"):
        if images.numel() == 0: continue
        images = images.to(device)
        with torch.cuda.amp.autocast():
            outputs = torch.sigmoid(model(images))
        val_preds.append(outputs.cpu().numpy())
        val_targets.append(labels.numpy())

val_preds = np.vstack(val_preds)
val_targets = np.vstack(val_targets)

best_thresholds = []
print("\n--- SOGLIE OTTIME CALCOLATE (INDICE DI YOUDEN) ---")
for j, c in enumerate(classes):
    fpr, tpr, thresholds = roc_curve(val_targets[:, j], val_preds[:, j])
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_thresh = thresholds[best_idx]
    if best_thresh > 0.95 or best_thresh < 0.05:
        best_thresh = 0.50
    best_thresholds.append(best_thresh)
    print(f"-> {c:<20} : Soglia Ottima = {best_thresh:.4f}")

# 2. INFERENZA SUL TEST SET
print("\nEsecuzione inferenza sul Test Set...")
test_preds, test_targets = [], []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Test Inference"):
        if images.numel() == 0: continue
        images = images.to(device)
        with torch.cuda.amp.autocast():
            outputs = torch.sigmoid(model(images))
        test_preds.append(outputs.cpu().numpy())
        test_targets.append(labels.numpy())

test_preds = np.vstack(test_preds)
test_targets = np.vstack(test_targets)

test_preds_bin = np.zeros_like(test_preds)
for j in range(num_classes):
    test_preds_bin[:, j] = (test_preds[:, j] >= best_thresholds[j]).astype(int)

test_aucs = []
for j in range(num_classes):
    try: test_aucs.append(roc_auc_score(test_targets[:, j], test_preds[:, j]))
    except ValueError: test_aucs.append(0.5)

print("\n" + "="*60)
print(f"➔ VECCHIO RECORD MACRO ROC-AUC (Grid Search): 0.8512")
print(f"➔ NUOVO MACRO ROC-AUC (CON LE SOGLIE CORRETTE): {np.mean(test_aucs):.4f}")
print("="*60)
print("\nClassification Report Bilanciato e Corretto:")
print(classification_report(test_targets.astype(int), test_preds_bin.astype(int), target_names=classes, zero_division=0))

# =====================================================
# AGGIUNTA TABELLE DETTAGLIO E 6 METRICHE GLOBALI
# =====================================================
print("\n" + "="*80)
print("TABELLA 1: DETTAGLIO PER CLASSE (Feature Fusion Ensemble)")
print("="*80)
print(f"{'Patologia':<20} | {'Soglia':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}")
print("-" * 80)

class_pr, class_rc, class_f1, _ = precision_recall_fscore_support(
    test_targets.astype(int), test_preds_bin.astype(int), average=None, zero_division=0
)

pr_auc_list = []
class_auc_dict = {}
class_pr_dict = {}
class_rc_dict = {}
class_f1_dict = {}

for j, c in enumerate(classes):
    try:
        roc_c = roc_auc_score(test_targets[:, j], test_preds[:, j])
    except ValueError:
        roc_c = 0.5000
    
    try:
        pr_c = average_precision_score(test_targets[:, j], test_preds[:, j])
    except ValueError:
        pr_c = 0.0000
        
    pr_auc_list.append(pr_c)
    class_auc_dict[c] = roc_c
    class_pr_dict[c] = class_pr[j]
    class_rc_dict[c] = class_rc[j]
    class_f1_dict[c] = class_f1[j]
    
    print(f"{c:<20} | {best_thresholds[j]:<8.4f} | {roc_c:<8.4f} | {pr_c:<8.4f} | {class_pr[j]:<9.4f} | {class_rc[j]:<8.4f} | {class_f1[j]:<8.4f}")

# Calcolo delle 6 metriche globali richieste
macro_roc_auc = np.mean(list(class_auc_dict.values()))
micro_roc_auc = roc_auc_score(test_targets.astype(int), test_preds, average='micro')
macro_pr_auc = np.mean(pr_auc_list)
macro_precision = np.mean(list(class_pr_dict.values()))
macro_recall = np.mean(list(class_rc_dict.values()))
macro_f1 = np.mean(list(class_f1_dict.values()))

print("\n" + "="*50)
print("TABELLA 2: LE 6 METRICHE GLOBALI (Feature Fusion)")
print("="*50)
print(f"{'Metrica Globale':<30} | {'Valore':<10}")
print("-" * 43)
print(f"{'Media Macro ROC-AUC':<30} | {macro_roc_auc:<10.4f}")
print(f"{'Media Micro ROC-AUC':<30} | {micro_roc_auc:<10.4f}")
print(f"{'Macro PR-AUC':<30} | {macro_pr_auc:<10.4f}")
print(f"{'Macro Precision':<30} | {macro_precision:<10.4f}")
print(f"{'Macro Recall':<30} | {macro_recall:<10.4f}")
print(f"{'Macro F1-Score':<30} | {macro_f1:<10.4f}")
print("="*50)