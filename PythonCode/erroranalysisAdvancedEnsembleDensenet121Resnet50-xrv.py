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
from sklearn.metrics import multilabel_confusion_matrix, roc_curve
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

# DATASET PIPELINE
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

# ARCHITETTURA (Pulita)
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
model.load_state_dict(torch.load(saved_ensemble_path, map_location=device))
model.eval()

val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)
test_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

val_loader = DataLoader(NIHChestDatasetXRV(val_df, test_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)
test_loader = DataLoader(NIHChestDatasetXRV(test_df, test_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

# 1. CALCOLO SOGLIE OTTIME (Youden su Validation)
print("Calcolo soglie ottime sul Validation Set...")
val_preds, val_targets = [], []
with torch.no_grad():
    for images, labels in val_loader:
        if images.numel() == 0: continue
        images = images.to(device)
        with torch.cuda.amp.autocast():
            outputs = torch.sigmoid(model(images))
        val_preds.append(outputs.cpu().numpy())
        val_targets.append(labels.numpy())

val_preds = np.vstack(val_preds)
val_targets = np.vstack(val_targets)

best_thresholds = []
for j in range(num_classes):
    fpr, tpr, thresholds = roc_curve(val_targets[:, j], val_preds[:, j])
    best_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[best_idx]
    if best_thresh > 0.95 or best_thresh < 0.05: best_thresh = 0.50
    best_thresholds.append(best_thresh)

# 2. INFERENZA SUL TEST SET
print("Esecuzione inferenza sul Test Set per Error Analysis...")
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

# Binarizzazione con soglie ottime
test_preds_bin = np.zeros_like(test_preds)
for j in range(num_classes):
    test_preds_bin[:, j] = (test_preds[:, j] >= best_thresholds[j]).astype(int)

# 3. ANALISI DEGLI ERRORI AVANZATA
print("\n" + "="*60)
print(" DETTAGLIO ERRORI DI CLASSIFICAZIONE (CON SOGLIE DI YOUDEN)")
print("="*60)

# Calcoliamo le matrici di confusione multilabel
mcm = multilabel_confusion_matrix(test_targets.astype(int), test_preds_bin.astype(int))

error_data = []

for j, c in enumerate(classes):
    tn, fp, fn, tp = mcm[j].ravel()
    
    total_positives = tp + fn
    total_negatives = tn + fp
    
    # Rapporti di errore
    false_positive_rate = fp / total_negatives if total_negatives > 0 else 0
    false_negative_rate = fn / total_positives if total_positives > 0 else 0
    
    print(f"\nPATOLOGIA: {c.upper()}")
    print(f"  [-] Veri Negativi (TN): {tn:<5} | [+] Veri Positivi (TP): {tp}")
    print(f"  [X] Falsi Positivi (FP): {fp:<5} -> (Sani scambiati per malati)")
    print(f"  [X] Falsi Negativi (FN): {fn:<5} -> (Malati PERSI dal modello!)")
    print(f"  --> Tasso Falsi Positivi (FPR): {false_positive_rate*100:.2f}%")
    print(f"  --> Tasso Falsi Negativi (FNR): {false_negative_rate*100:.2f}%")
    
    error_data.append({
        "Classe": c,
        "Soglia_Youden": best_thresholds[j],
        "Veri_Negativi": tn,
        "Falsi_Positivi": fp,
        "Falsi_Negativi": fn,
        "Veri_Positivi": tp,
        "Tasso_Falsi_Positivi_%": round(false_positive_rate*100, 2),
        "Tasso_Falsi_Negativi_%": round(false_negative_rate*100, 2)
    })

# Salvataggio in CSV per analisi grafica successiva
df_errors = pd.DataFrame(error_data)
df_errors.to_csv("PythonCode/error_analysis_report.csv", index=False)
print("\n" + "="*60)
print("✔ Report di Error Analysis salvato in: PythonCode/error_analysis_report.csv")
print("="*60)