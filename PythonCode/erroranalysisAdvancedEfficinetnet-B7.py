import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# =====================================================
# CONFIGURAZIONE PATHS
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
path_pesi_b7 = "/home/gpuvm/Desktop/Luca Migliaccio/best_efficientnet_b7_asl.pth"

IMAGE_SIZE = 600  # Risoluzione nativa della B7
BATCH_SIZE = 16

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)

# =====================================================
# HARDCODING DELLE SOGLIE DI YOUDEN REALI DELLA B7
# =====================================================
# Inseriamo esattamente le soglie sputate dal tuo ultimo run della B7
b7_thresholds = {
    "Atelectasis": 0.4353, "Cardiomegaly": 0.3687, "Effusion": 0.4973,
    "Infiltration": 0.5244, "Mass": 0.4429, "Nodule": 0.4465,
    "Pneumonia": 0.3818, "Pneumothorax": 0.4033, "Consolidation": 0.4021,
    "Edema": 0.3894, "Emphysema": 0.3718, "Fibrosis": 0.3796,
    "Pleural_Thickening": 0.4299, "Hernia": 0.3140
}
print("Soglie di Youden ottimizzate per la EfficientNet-B7 caricate con successo! ✔")

# =====================================================
# UTILS & DATASET (Ricalcato sul tuo con modifiche RGB per B7)
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

class NIHChestDatasetB7(Dataset):
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
            # EfficientNet-B7 richiede rigorosamente 3 canali RGB
            image = Image.open(img_path).convert("RGB")
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
# INIZIALIZZAZIONE MODELLO E CARICAMENTO PESI B7
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n[INFO] Caricamento EfficientNet-B7 in memoria...")
model = models.efficientnet_b7(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(path_pesi_b7, map_location=device))
model = model.to(device)
model.eval()

print("EfficientNet-B7 pronta per l'Error Analysis! ✔")

# =====================================================
# PIPELINE DI INFERENZA E TRACCIAMENTO ERRORI
# =====================================================
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalizzazione obbligatoria B7
])

test_df = pd.read_csv(test_csv)
test_loader = DataLoader(
    NIHChestDatasetB7(test_df, test_transform), 
    batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn
)

error_records = []
print("\nAnalisi e tracciamento degli errori sul Test Set della EfficientNet-B7...")

with torch.no_grad():
    for images, labels, names in tqdm(test_loader, desc="Analyzing B7 Errors"):
        if images.numel() == 0: continue
        images = images.to(device)
        
        # Sfruttiamo il nuovo formato autocast raccomandato da PyTorch
        with torch.amp.autocast('cuda'):
            outputs = torch.sigmoid(model(images))
            
        probs_b7 = outputs.cpu().numpy()
        targets = labels.numpy()
        
        for idx, img_name in enumerate(names):
            for j, c in enumerate(classes):
                prob = float(probs_b7[idx, j])
                target = int(targets[idx, j])
                thresh = b7_thresholds[c]
                pred = 1 if prob >= thresh else 0
                
                # Se la predizione non coincide con il target reale, registriamo l'errore
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
csv_save_path = "checkpoints/error_analysis_efficientnet_b7.csv"
df_errors.to_csv(csv_save_path, index=False)

print(f"\nAnalisi completata! Trovati {len(df_errors)} errori totali multiclasse per la EfficientNet-B7.")
print(f"I dati grezzi di errore sono stati salvati in: {csv_save_path} ✔")

# =====================================================
# ESTRAZIONE REPORT CRITICO TOP 5 FN / TOP 5 FP
# =====================================================
df_fn = df_errors[df_errors["Type"] == "FN"].sort_values(by="Error_Margin", ascending=False).head(5)
df_fp = df_errors[df_errors["Type"] == "FP"].sort_values(by="Error_Margin", ascending=False).head(5)

print("\n" + "="*70)
print("--- TOP 5 PEGGIORI FALSI NEGATIVI DELLA EFFICIENTNET-B7 (Mancati) ---")
print("="*70)
if not df_fn.empty:
    print(df_fn.to_string(index=False))
else:
    print("Nessun falso negativo trovato.")

print("\n" + "="*70)
print("--- TOP 5 PEGGIORI FALSI POSITIVI DELLA EFFICIENTNET-B7 (Falsi Allarmi) ---")
print("="*70)
if not df_fp.empty:
    print(df_fp.to_string(index=False))
else:
    print("Nessun falso positivo trovato.")