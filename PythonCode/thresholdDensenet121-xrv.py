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
from sklearn.metrics import f1_score
import torchxrayvision as xrv

# =====================================================
# CONFIGURAZIONE PATHS (Allineati alla V4)
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
val_csv   = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"

# Caricamento dal corretto path dei checkpoint della v4
model_path = "checkpoints/best_densenet121_v4_xrv.pth" 

# =====================================================
# COSTANTI
# =====================================================
IMAGE_SIZE = 512  # Risoluzione a 512px

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)

# =====================================================
# UTILS
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

# DATASET ADATTATO PER TORCHXRAYVISION (Copiato dal tuo train_v4)
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

        # XRV richiede scala di grigi (L)
        image = Image.open(img_path).convert("L")
        
        # Conversione e normalizzazione custom richiesta dai pesi XRV [-1024, 1024]
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
# DATA LOADER (Senza normalizzazione ImageNet!)
# =====================================================
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor() # 1 canale nativo
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_df = pd.read_csv(val_csv)
val_loader = DataLoader(
    NIHChestDatasetXRV(val_df, val_transform), 
    batch_size=16, 
    shuffle=False, 
    num_workers=2, 
    pin_memory=True, 
    collate_fn=collate_fn
)

# =====================================================
# DEFINIZIONE E CARICAMENTO ARCHITETTURA CUSTOM V4
# =====================================================
base_model = xrv.models.DenseNet(weights="densenet121-res224-all")

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

model = XRVFeatureExtractor(base_model, num_classes)
# Carichiamo lo state dict salvato stanotte
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
print(f"\nModello v4 (XRV + 512px) caricato con successo da: {model_path} ✔")

# =====================================================
# INFERENCE (Estrazione predizioni pulite)
# =====================================================
all_probs = []
all_targets = []

print("\nEstrazione delle predizioni sul Validation Set (Formato XRV)...")
with torch.no_grad():
    for images, labels in tqdm(val_loader):
        if images.numel() == 0: continue
        images = images.to(device)
        
        # Utilizziamo autocast come nel loop di train per consistenza matematica
        with torch.cuda.amp.autocast():
            outputs = model(images)
            
        probs = torch.sigmoid(outputs)
        all_probs.append(probs.cpu())
        all_targets.append(labels.cpu())

all_probs = torch.cat(all_probs).numpy()
all_targets = torch.cat(all_targets).numpy()

# =====================================================
# THRESHOLD OPTIMIZATION (F1-Score Tuning)
# =====================================================
best_thresholds = {}
print("\nOttimizzazione delle soglie per singola classe (Max F1)...")

for j, c in enumerate(classes):
    best_f1 = 0
    best_t = 0.5
    
    # Scansione dei possibili valori di soglia da 0.05 a 0.95
    for t in np.arange(0.05, 0.95, 0.05):
        preds = (all_probs[:, j] >= t).astype(int)
        f1 = f1_score(all_targets[:, j], preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    best_thresholds[c] = float(best_t)
    print(f"  -> Class: {c:<20} | Nuova Soglia: {best_t:.2f} | F1-Score Val: {best_f1:.4f}")

# Salvataggio del nuovo file JSON delle soglie
save_path = "checkpoints/optimized_thresholds_v4_xrv.json"
with open(save_path, "w") as f:
    json.dump(best_thresholds, f, indent=4)

print(f"\nNuove soglie v4 salvate con successo in: {save_path} ✔")