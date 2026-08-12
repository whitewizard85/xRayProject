import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torchxrayvision as xrv

# =====================================================
# CONFIGURAZIONE PATHS
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv" 
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"

model_path_v4 = "checkpoints/best_densenet121_v4_xrv.pth"
model_path_v5 = "checkpoints/best_resnet50_v5_xrv.pth"

IMAGE_SIZE = 512
classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)

# =====================================================
# DATASET & INFERENCE SYSTEM
# =====================================================
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Caricamento Reti
base_densenet = xrv.models.DenseNet(weights="densenet121-res224-all")
class XRVFeatureExtractor(nn.Module):
    def __init__(self, xrv_model, num_classes):
        super().__init__()
        self.features = xrv_model.features           
        self.classifier = nn.Linear(1024, num_classes) 
    def forward(self, x):
        out = F.relu(self.features(x), inplace=True)
        return self.classifier(torch.flatten(F.adaptive_avg_pool2d(out, (1, 1)), 1))

model_v4 = XRVFeatureExtractor(base_densenet, num_classes).to(device)
model_v4.load_state_dict(torch.load(model_path_v4, map_location=device))
model_v4.eval()

base_resnet = xrv.models.ResNet(weights="resnet50-res512-all")
class XRVResNetFeatureExtractor(nn.Module):
    def __init__(self, xrv_resnet, num_classes):
        super().__init__()
        self.base_resnet = xrv_resnet
        self.classifier = nn.Linear(2048, num_classes)
    def forward(self, x): return self.classifier(self.base_resnet.features(x))

model_v5 = XRVResNetFeatureExtractor(base_resnet, num_classes).to(device)
model_v5.load_state_dict(torch.load(model_path_v5, map_location=device))
model_v5.eval()

def extract_probabilities(csv_path, desc):
    df = pd.read_csv(csv_path)
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    loader = DataLoader(NIHChestDatasetXRV(df, transform), batch_size=16, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    probs_v4_all, probs_v5_all, targets_all = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            if images.numel() == 0: continue
            images = images.to(device)
            with torch.cuda.amp.autocast():
                out_v4 = model_v4(images)
                out_v5 = model_v5(images)
            probs_v4_all.append(torch.sigmoid(out_v4).cpu().numpy())
            probs_v5_all.append(torch.sigmoid(out_v5).cpu().numpy())
            targets_all.append(labels.numpy())
            
    # SCUDO PROTETTIVO: Forziamo il cast a float32 per evitare il crash di SciPy
    return (np.vstack(probs_v4_all).astype(np.float32), 
            np.vstack(probs_v5_all).astype(np.float32), 
            np.vstack(targets_all).astype(np.float32))

# --- 1. Estrazione Dati ---
print("1. Estrazione predizioni dal VALIDATION SET (servirà come Training Set per lo Stacking)...")
val_v4, val_v5, val_targets = extract_probabilities(val_csv, "Validation Inference")

print("\n2. Estrazione predizioni dal TEST SET (per la verifica finale cieca)...")
test_v4, test_v5, test_targets = extract_probabilities(test_csv, "Test Inference")

# --- 2. Costruzione delle Feature dello Stacking ---
X_train_meta = np.hstack([val_v4, val_v5])
X_test_meta = np.hstack([test_v4, test_v5])

# Assicuriamoci che anche i target siano interi per il classificatore
val_targets_int = val_targets.astype(np.int32)
test_targets_int = test_targets.astype(np.int32)

# =====================================================
# TUNING TRAMITE STRATEGIA DI STACKING
# =====================================================
print("\n" + "="*60)
print("ADDESTRAMENTO DEI META-CLASSIFICATORI (RANDOM FOREST) PER CLASSE")
print("="*60)

meta_models = {}
final_test_probs = np.zeros_like(test_v4)
final_test_preds_bin = np.zeros_like(test_v4)
test_auc_list = []

for j, c in enumerate(classes):
    y_train_class = val_targets_int[:, j]
    y_test_class = test_targets_int[:, j]
    
    # Random Forest leggero e robusto contro l'overfitting
    meta_clf = RandomForestClassifier(
        n_estimators=150, 
        max_depth=5, 
        min_samples_leaf=4, 
        random_state=42, 
        n_jobs=-1
    )
    
    # Train sul Validation Set
    meta_clf.fit(X_train_meta, y_train_class)
    meta_models[c] = meta_clf
    
    # Predizione sul Test Set (probabilità della classe positiva)
    probs_test_class = meta_clf.predict_proba(X_test_meta)[:, 1]
    final_test_probs[:, j] = probs_test_class
    final_test_preds_bin[:, j] = (probs_test_class >= 0.50).astype(int)
    
    try:
        auc_t = roc_auc_score(y_test_class, probs_test_class)
        test_auc_list.append(auc_t)
        print(f"-> {c:<20} | Stacking Meta-AUC sul Test Set: {auc_t:.4f}")
    except ValueError:
        test_auc_list.append(0.5)
        print(f"-> {c:<20} | Stacking Meta-AUC sul Test Set: Errore (Classe assente)")

# =====================================================
# REPORT FINALE COMPARATIVO
# =====================================================
print("\n" + "="*60)
print("REPORT FINALE SUL TEST SET CON META-ENSEMBLE STACKING")
print("="*60)

print(classification_report(test_targets_int, final_test_preds_bin, target_names=classes, zero_division=0))

stacking_macro_auc = np.mean(test_auc_list)
print("-"*60)
print(f"➔ VECCHIO MACRO AVERAGE ROC-AUC (Media Semplice):  0.8499")
print(f"➔ PRECEDENTE RECORD ROC-AUC (Grid Search):          0.8512")
print(f"➔ NUOVO MACRO AVERAGE ROC-AUC (STACKING RF):       {stacking_macro_auc:.4f}")
print("-"*60)

# Salvataggio
output_dir = "checkpoints"
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "stacking_test_probs.npy"), final_test_probs)

print("[SUCCESS] Tuning Stacking completato senza conflitti di tipo! 🚀")