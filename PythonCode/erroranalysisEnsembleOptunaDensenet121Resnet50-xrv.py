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
import torchxrayvision as xrv

# =====================================================
# CONFIGURAZIONE PATHS
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"

model_path_v4 = "checkpoints/best_densenet121_v4_xrv.pth"
model_path_v5 = "checkpoints/best_resnet50_v5_xrv.pth"

# Carichiamo la configurazione iper-ottimizzata generata da Optuna
optuna_config_path = "checkpoints/hyperoptimized_ensemble_config.json"

IMAGE_SIZE = 512

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)

# =====================================================
# UTILS & DATASET (Blindato contro i leak di PIL Image)
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
        self.fallback_to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        label_str = row["Finding Labels"]

        img_path = get_image_path(img_name)
        if img_path is None:
            return None, None, None

        try:
            image = Image.open(img_path).convert("L")
            
            # Normalizzazione custom richiesta dai pesi XRV
            img_np = np.array(image)
            img_np = xrv.datasets.normalize(img_np, maxval=255)
            image = Image.fromarray(img_np)

            label = encode_labels(label_str)

            if self.transform:
                image = self.transform(image)
            
            if not isinstance(image, torch.Tensor):
                image = self.fallback_to_tensor(image)

            return image, label, img_name
            
        except Exception:
            return None, None, None

def collate_fn(batch):
    batch = [b for b in batch if b is not None and b[0] is not None and b[1] is not None]
    if len(batch) == 0: 
        return torch.empty(0), torch.empty(0), []
    
    cleaned_images = []
    for b in batch:
        img = b[0]
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        cleaned_images.append(img)
        
    images = torch.stack(cleaned_images)
    labels = torch.stack([b[1] for b in batch])
    img_names = [b[2] for b in batch]
    return images, labels, img_names

# =====================================================
# LOAD PARAMETRI DA JSON OPTUNA
# =====================================================
if not os.path.exists(optuna_config_path):
    raise FileNotFoundError(f"⚠️ Impossibile trovare il file {optuna_config_path}. Esegui prima lo script di Optuna!")

with open(optuna_config_path, "r") as f:
    optuna_data = json.load(f)

best_alphas = optuna_data["best_alphas_per_class"]
optimized_thresholds = optuna_data["optimized_thresholds_per_class"]
print("Parametri iper-ottimizzati di OPTUNA caricati con successo! 🚀")

# =====================================================
# ARCHITETTURE & CARICAMENTO MODELLI
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modello V4 (DenseNet121)
base_densenet = xrv.models.DenseNet(weights="densenet121-res224-all")
class XRVFeatureExtractor(nn.Module):
    def __init__(self, xrv_model, num_classes):
        super(XRVFeatureExtractor, self).__init__()
        self.features = xrv_model.features           
        self.classifier = nn.Linear(1024, num_classes) 
    def forward(self, x):
        out = F.relu(self.features(x), inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return self.classifier(torch.flatten(out, 1))

model_v4 = XRVFeatureExtractor(base_densenet, num_classes).to(device)
model_v4.load_state_dict(torch.load(model_path_v4, map_location=device))
model_v4.eval()

# Modello V5 (ResNet50)
base_resnet = xrv.models.ResNet(weights="resnet50-res512-all")
class XRVResNetFeatureExtractor(nn.Module):
    def __init__(self, xrv_resnet, num_classes):
        super(XRVResNetFeatureExtractor, self).__init__()
        self.base_resnet = xrv_resnet
        self.classifier = nn.Linear(2048, num_classes)
    def forward(self, x):
        return self.classifier(self.base_resnet.features(x))

model_v5 = XRVResNetFeatureExtractor(base_resnet, num_classes).to(device)
model_v5.load_state_dict(torch.load(model_path_v5, map_location=device))
model_v5.eval()

print("Entrambi i modelli sono pronti per l'analisi del nuovo Optuna Ensemble! ✔")

# =====================================================
# INFERENCE E FILTRAGGIO ERRORI CONGIUNTI
# =====================================================
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

test_df = pd.read_csv(test_csv)
test_loader = DataLoader(
    NIHChestDatasetXRV(test_df, test_transform), 
    batch_size=16, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn
)

error_records = []
print("\nAnalisi degli errori sul Test Set basata sul nuovo Optuna Ensemble...")

with torch.no_grad():
    for images, labels, names in tqdm(test_loader, desc="Analyzing Optuna Errors"):
        if images.numel() == 0: continue
        images = images.to(device)
        
        with torch.cuda.amp.autocast():
            out_v4 = model_v4(images)
            out_v5 = model_v5(images)
            
        probs_v4 = torch.sigmoid(out_v4).cpu().numpy()
        probs_v5 = torch.sigmoid(out_v5).cpu().numpy()
        targets = labels.numpy()
        
        for idx, img_name in enumerate(names):
            for j, c in enumerate(classes):
                # Recuperiamo l'alfa e la soglia specifici estratti da Optuna per questa classe
                alpha = best_alphas[c]
                thresh = optimized_thresholds[c]
                
                # Late Fusion Pesata: calcolo della probabilità calibrata
                prob = float(alpha * probs_v4[idx, j] + (1.0 - alpha) * probs_v5[idx, j])
                target = int(targets[idx, j])
                pred = 1 if prob >= thresh else 0
                
                if pred != target:
                    err_type = "FN" if target == 1 else "FP"
                    err_margin = float(thresh - prob) if err_type == "FN" else float(prob - thresh)
                    
                    error_records.append({
                        "Image_Index": img_name,
                        "Class": c,
                        "Type": err_type,
                        "Probability": prob,
                        "Threshold": thresh,
                        "Target": target,
                        "Error_Margin": err_margin
                    })

df_errors = pd.DataFrame(error_records)
os.makedirs("checkpoints", exist_ok=True)
csv_save_path = "checkpoints/error_analysis_optuna_ensemble.csv"
df_errors.to_csv(csv_save_path, index=False)

print(f"\nAnalisi completata! Trovati {len(df_errors)} errori totali multiclasse con Optuna.")
print(f"I dati grezzi di Optuna sono salvati in: {csv_save_path} ✔")

# =====================================================
# EXTRACTION DEI TOP 5 FN / TOP 5 FP (OPTUNA EDITION)
# =====================================================
df_fn = df_errors[df_errors["Type"] == "FN"].sort_values(by="Error_Margin", ascending=False).head(5)
df_fp = df_errors[df_errors["Type"] == "FP"].sort_values(by="Error_Margin", ascending=False).head(5)

print("\n" + "="*70)
print("--- TOP 5 PEGGIORI FALSI NEGATIVI CON OPTUNA ENSEMBLE ---")
print("="*70)
if not df_fn.empty:
    print(df_fn.to_string(index=False))
else:
    print("Nessun falso negativo trovato.")

print("\n" + "="*70)
print("--- TOP 5 PEGGIORI FALSI POSITIVI CON OPTUNA ENSEMBLE ---")
print("="*70)
if not df_fp.empty:
    print(df_fp.to_string(index=False))
else:
    print("Nessun falso positivo trovato.")