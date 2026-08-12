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
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score

# --- CONFIGURAZIONE ---
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
saved_ensemble_path = "checkpoints/best_feature_fusion_ensemble.pth"

IMAGE_SIZE = 512
BATCH_SIZE = 16
classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", 
           "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
num_classes = len(classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DEFINIZIONE MODELLO ---
class FeatureFusionEnsemble(nn.Module):
    def __init__(self, num_classes):
        super(FeatureFusionEnsemble, self).__init__()
        import torchxrayvision as xrv # Import qui per sicurezza
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
        if len(f_dn.shape) > 2: f_dn = F.adaptive_avg_pool2d(f_dn, (1, 1)); f_dn = torch.flatten(f_dn, 1)
        f_rn = self.rn_model.features(x)
        if len(f_rn.shape) > 2: f_rn = F.adaptive_avg_pool2d(f_rn, (1, 1)); f_rn = torch.flatten(f_rn, 1)
        return self.fusion_classifier(torch.cat((f_dn, f_rn), dim=1))

# --- DATASET & UTILS ---
def get_image_path(img_name):
    for i in range(1, 13):
        path = os.path.join(root_dir, f"images_{i:03d}", "images", img_name)
        if os.path.exists(path): return path
    return None

class NIHChestDatasetXRV(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True); self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]; img_path = get_image_path(row["Image Index"])
        if img_path is None: return None, None
        try:
            import torchxrayvision as xrv
            image = Image.open(img_path).convert("L")
            img_np = xrv.datasets.normalize(np.array(image), maxval=255)
            image = Image.fromarray(img_np)
            vec = torch.zeros(num_classes)
            for l in str(row["Finding Labels"]).split("|"):
                if l in classes: vec[classes.index(l)] = 1.0
            if self.transform: image = self.transform(image)
            return image, vec
        except Exception: return None, None

def collate_fn(batch):
    batch = [b for b in batch if b is not None and b[0] is not None]
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]) if batch else (torch.empty(0), torch.empty(0))

# --- ESECUZIONE ---
model = FeatureFusionEnsemble(num_classes).to(device)
model.load_state_dict(torch.load(saved_ensemble_path, map_location=device))
model.eval()

transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
val_loader = DataLoader(NIHChestDatasetXRV(pd.read_csv(val_csv), transform), batch_size=BATCH_SIZE, num_workers=4, collate_fn=collate_fn)
test_loader = DataLoader(NIHChestDatasetXRV(pd.read_csv(test_csv), transform), batch_size=BATCH_SIZE, num_workers=4, collate_fn=collate_fn)

# 1. VALIDAZIONE (Soglie)
val_preds, val_targets = [], []
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Validazione"):
        if images.numel() == 0: continue
        val_preds.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
        val_targets.append(labels.numpy())

val_preds, val_targets = np.vstack(val_preds), np.vstack(val_targets)
best_thresholds = []
for j in range(num_classes):
    p, r, t = precision_recall_curve(val_targets[:, j], val_preds[:, j])
    f1 = 2 * (p * r) / (p + r + 1e-16)
    best_thresholds.append(t[np.argmax(f1)] if np.argmax(f1) < len(t) else 0.5)

# 2. TEST (Inferenza)
test_preds, test_targets = [], []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Test Inference"):
        if images.numel() == 0: continue
        test_preds.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
        test_targets.append(labels.numpy())

test_preds, test_targets = np.vstack(test_preds), np.vstack(test_targets)
test_preds_bin = (test_preds >= np.array(best_thresholds)).astype(int)

# 3. REPORT FINALE
print("\n" + "="*60)
print("REPORT DI CLASSIFICAZIONE (TEST SET)")
print("="*60)
print(classification_report(test_targets, test_preds_bin, target_names=classes, zero_division=0))

# Calcolo Macro ROC-AUC
auc_scores = [roc_auc_score(test_targets[:, j], test_preds[:, j]) for j in range(num_classes)]
print(f"\n➔ MACRO ROC-AUC (TEST SET): {np.mean(auc_scores):.4f}")
print("="*60)