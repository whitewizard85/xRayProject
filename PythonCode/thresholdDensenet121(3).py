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
from sklearn.metrics import f1_score

# =====================================================
# CONFIGURAZIONE PATHS (Allineati al tuo script)
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
train_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv"
val_csv   = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"

# Caricamento diretto dalla cartella principale come fa il tuo script di train
model_path = "best_densenet121_v3_512px_asl.pth" 

# =====================================================
# COSTANTI
# =====================================================
IMAGE_SIZE = 512  # Risoluzione v3 a 512px

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
        if img_path is None: return None, None
        image = Image.open(img_path).convert("RGB")
        label = encode_labels(label_str)
        if self.transform: image = self.transform(image)
        return image, label

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0: return torch.empty(0), torch.empty(0)
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return images, labels

# =====================================================
# DATA LOADER
# =====================================================
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_df = pd.read_csv(val_csv)
val_loader = DataLoader(
    NIHChestDataset(val_df, val_transform), 
    batch_size=16, 
    shuffle=False, 
    num_workers=2, 
    pin_memory=True, 
    collate_fn=collate_fn
)

# =====================================================
# LOAD MODEL
# =====================================================
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
print(f"\nModello v3 caricato con successo da: {model_path} ✔")

# =====================================================
# INFERENCE
# =====================================================
all_probs = []
all_targets = []

print("Estrazione delle predizioni sul Validation Set (512px)...")
with torch.no_grad():
    for images, labels in tqdm(val_loader):
        if images.numel() == 0: continue
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        all_probs.append(probs.cpu())
        all_targets.append(labels.cpu())

all_probs = torch.cat(all_probs).numpy()
all_targets = torch.cat(all_targets).numpy()

# =====================================================
# THRESHOLD OPTIMIZATION
# =====================================================
best_thresholds = {}
print("\nOttimizzazione delle soglie per singola classe (Max F1)...")

for j, c in enumerate(classes):
    best_f1 = 0
    best_t = 0.5
    
    for t in np.arange(0.05, 0.95, 0.05):
        preds = (all_probs[:, j] >= t).astype(int)
        f1 = f1_score(all_targets[:, j], preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    best_thresholds[c] = float(best_t)
    print(f"  -> Class: {c:<20} | Nuova Soglia: {best_t:.2f} | F1-Score Val: {best_f1:.4f}")

# Salvataggio del nuovo file JSON delle soglie
save_path = "optimized_thresholds_v3.json"
with open(save_path, "w") as f:
    json.dump(best_thresholds, f, indent=4)

print(f"\nNuove soglie salvate con successo in: {save_path} ✔")