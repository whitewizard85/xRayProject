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
# CONFIGURAZIONE PATHS (Allineati alla V5)
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"

model_path = "checkpoints/best_resnet50_v5_xrv.pth"
thresholds_path = "checkpoints/optimized_thresholds_v5_xrv.json"
output_error_csv = "checkpoints/error_analysis_v5_xrv.csv"

# =====================================================
# COSTANTI & CLASSI
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
# UTILS & DATASET (Specifico per XRV)
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
            return None, None, None

        # Scala di grigi tassativa per XRV
        image = Image.open(img_path).convert("L")
        
        # Normalizzazione custom richiesta dai pesi XRV [-1024, 1024]
        img_np = np.array(image)
        img_np = xrv.datasets.normalize(img_np, maxval=255)
        image = Image.fromarray(img_np)

        label = encode_labels(label_str)

        if self.transform:
            image = self.transform(image)

        return image, label, img_name

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0: return torch.empty(0), torch.empty(0), []
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    names = [b[2] for b in batch]
    return images, labels, names

# =====================================================
# LOAD DATA & THRESHOLDS
# =====================================================
print(f"Caricamento soglie ottimizzate v5 da: {thresholds_path}")
with open(thresholds_path, "r") as f:
    optimized_thresholds = json.load(f)

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor() # 1 canale
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_df = pd.read_csv(test_csv)
test_loader = DataLoader(
    NIHChestDatasetXRV(test_df, test_transform),
    batch_size=16, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn
)

# =====================================================
# LOAD ARCHITETTURA RESNET50 V5
# =====================================================
base_model = xrv.models.ResNet(weights="resnet50-res512-all").to(device)

class XRVResNetFeatureExtractor(nn.Module):
    def __init__(self, xrv_resnet, num_classes):
        super(XRVResNetFeatureExtractor, self).__init__()
        self.base_resnet = xrv_resnet
        self.classifier = nn.Linear(2048, num_classes) # Input a 2048 specifico di ResNet50 di XRV

    def forward(self, x):
        features = self.base_resnet.features(x)
        out = self.classifier(features)
        return out

model = XRVResNetFeatureExtractor(base_model, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
print(f"\nModello v5 (ResNet50 XRV @ 512px) caricato con successo per l'Error Analysis! ✔")

# =====================================================
# INFERENCE & ERROR TRACKING
# =====================================================
records = []

print("\nEstrazione errori sul Test Set...")
with torch.no_grad():
    for images, labels, names in tqdm(test_loader, desc="Analyzing Errors"):
        if images.numel() == 0: continue
        images = images.to(device)
        
        # Coerenza con autocast a precisione mista usato in train
        with torch.cuda.amp.autocast():
            outputs = model(images)
            
        probs = torch.sigmoid(outputs).cpu().numpy().astype(np.float32) 
        targets = labels.cpu().numpy().astype(np.int32)
        
        for idx in range(len(names)):
            img_name = names[idx]
            img_probs = probs[idx]
            img_targets = targets[idx]
            
            for j, c in enumerate(classes):
                t = optimized_thresholds[c]
                prob = img_probs[j]
                target = img_targets[j]
                pred = 1.0 if prob >= t else 0.0
                
                # Identifica l'errore
                error_type = "NONE"
                error_margin = 0.0
                
                if pred == 1.0 and target == 0.0:
                    error_type = "FP"  # Falso Positivo (Sovra-diagnosi)
                    error_margin = float(prob - t)  # Confidenza d'errore in eccesso
                elif pred == 0.0 and target == 1.0:
                    error_type = "FN"  # Falso Negativo (Mancata diagnosi)
                    error_margin = float(t - prob)  # Distanza dal target ottimale
                
                if error_type != "NONE":
                    records.append({
                        "Image_Index": img_name,
                        "Class": c,
                        "Type": error_type,
                        "Probability": float(prob),
                        "Threshold": float(t),
                        "Target": int(target),
                        "Error_Margin": error_margin
                    })

# Creazione cartella checkpoints se non esiste e salvataggio
os.makedirs("checkpoints", exist_ok=True)
df_errors = pd.DataFrame(records)
df_errors.to_csv(output_error_csv, index=False)

print(f"\nAnalisi completata! Trovati {len(df_errors)} errori totali per singola classe.")
print(f"I dati sono stati salvati in: {output_error_csv} ✔")

# Mostra un'anteprima dei peggiori errori (Margine più alto)
print("\n" + "="*70)
print("--- TOP 5 PEGGIORI FALSI NEGATIVI (Malattier mancate clamorosamente) ---")
print("="*70)
if not df_errors[df_errors["Type"] == "FN"].empty:
    print(df_errors[df_errors["Type"] == "FN"].sort_values(by="Error_Margin", ascending=False).head(5).to_string(index=False))
else:
    print("Nessun Falso Negativo trovato.")

print("\n" + "="*70)
print("--- TOP 5 PEGGIORI FALSI POSITIVI (Allarmi finti con massima confidenza) ---")
print("="*70)
if not df_errors[df_errors["Type"] == "FP"].empty:
    print(df_errors[df_errors["Type"] == "FP"].sort_values(by="Error_Margin", ascending=False).head(5).to_string(index=False))
else:
    print("Nessun Falso Positivo trovato.")