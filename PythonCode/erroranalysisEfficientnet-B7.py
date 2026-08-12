import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# =====================================================
# CONFIGURAZIONE PATHS (Allineati alla tua VM)
# =====================================================
BASE_DIR = "/home/gpuvm/Desktop/Luca Migliaccio"
PYTHON_DIR = os.path.join(BASE_DIR, "PythonCode")
root_dir = os.path.join(BASE_DIR, "archive")

test_csv = os.path.join(PYTHON_DIR, "test_split.csv")
model_path = os.path.join(BASE_DIR, "best_efficientnet_b7.pth")
thresholds_path = os.path.join(PYTHON_DIR, "optimized_thresholds_EfficientNetB7.json")
output_error_csv = os.path.join(PYTHON_DIR, "error_analysis_EfficientNetB7.csv")

# =====================================================
# COSTANTI & CLASSI
# =====================================================
IMAGE_SIZE = 600  # Risoluzione corretta per EfficientNet-B7

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)

# =====================================================
# UTILS & DATASET
# =====================================================
def encode_labels(label_str):
    vec = torch.zeros(num_classes)
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
            return None, None, None
            
        image = Image.open(img_path).convert("RGB")
        label = encode_labels(label_str)
        
        if self.transform: 
            image = self.transform(image)
            
        return image, label, img_name

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0: 
        return torch.empty(0), torch.empty(0), []
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    names = [b[2] for b in batch]
    return images, labels, names

# =====================================================
# LOAD DATA & THRESHOLDS
# =====================================================
if not os.path.exists(thresholds_path):
    raise FileNotFoundError(f"File delle soglie non trovato: {thresholds_path}")

with open(thresholds_path, "r") as f:
    optimized_thresholds = json.load(f)

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_df = pd.read_csv(test_csv)
test_loader = DataLoader(
    NIHChestDataset(test_df, test_transform),
    batch_size=16, 
    shuffle=False, 
    num_workers=0,  # 0 per stabilità su VM
    pin_memory=True, 
    collate_fn=collate_fn
)

# =====================================================
# LOAD MODEL (Adattato per EfficientNet-B7)
# =====================================================
print("\nInizializzazione EfficientNet-B7...")
model = models.efficientnet_b7(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"File pesi non trovato: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
print(f"Modello EfficientNet-B7 caricato per l'Error Analysis ✔")

# =====================================================
# INFERENCE & ERROR TRACKING
# =====================================================
records = []

print("\nEstrazione errori sul Test Set...")
with torch.no_grad():
    for images, labels, names in tqdm(test_loader):
        if images.numel() == 0: 
            continue
        images = images.to(device)
        
        # Mixed precision coerente con il train
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            
        probs = torch.sigmoid(outputs).cpu().numpy()
        targets = labels.cpu().numpy()
        
        for idx in range(len(names)):
            img_name = names[idx]
            img_probs = probs[idx]
            img_targets = targets[idx]
            
            for j, c in enumerate(classes):
                t = optimized_thresholds[c]
                prob = img_probs[j]
                target = img_targets[j]
                pred = 1.0 if prob >= t else 0.0
                
                error_type = "NONE"
                error_margin = 0.0
                
                if pred == 1.0 and target == 0.0:
                    error_type = "FP"  # Falso Positivo
                    error_margin = float(prob - t)  # Grado di confidenza errata
                elif pred == 0.0 and target == 1.0:
                    error_type = "FN"  # Falso Negativo
                    error_margin = float(t - prob)  # Distanza dalla soglia minima
                
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

# Controllo se sono presenti anomalie registrate
if len(records) == 0:
    print("\nIncredibile! Nessun errore rilevato sul test set con le attuali configurazioni.")
else:
    # Salvataggio su disco del file CSV risultante
    df_errors = pd.DataFrame(records)
    df_errors.to_csv(output_error_csv, index=False)
    print(f"\nAnalisi completata! Trovati {len(df_errors)} errori totali distribuiti per classe.")
    print(f"I dettagli analitici sono stati salvati in: {output_error_csv} ✔")

    # Filtri di estrazione condizionale per l'anteprima a terminale
    fns = df_errors[df_errors["Type"] == "FN"]
    fps = df_errors[df_errors["Type"] == "FP"]

    print("\n--- TOP 5 PEGGIORI FALSI NEGATIVI (Malattie mancate clamorosamente) ---")
    if not fns.empty:
        print(fns.sort_values(by="Error_Margin", ascending=False).head(5).to_string(index=False))
    else:
        print("Nessun Falso Negativo registrato.")

    print("\n--- TOP 5 PEGGIORI FALSI POSITIVI (Allarmi finti con massima confidenza) ---")
    if not fps.empty:
        print(fps.sort_values(by="Error_Margin", ascending=False).head(5).to_string(index=False))
    else:
        print("Nessun Falso Positivo registrato.")