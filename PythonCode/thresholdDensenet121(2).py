import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.metrics import f1_score
from tqdm import tqdm

# =====================================================
# PATHS
# =====================================================
PYTHON_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode"
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

VAL_CSV = os.path.join(PYTHON_DIR, "val_split.csv")
MODEL_PATH = "best_densenet121_v2.pth"  # Il file salvato dal tuo script di train

# =====================================================
# CLASSES (Esattamente le 14 del tuo codice di train)
# =====================================================
classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)

# =====================================================
# LABEL ENCODING
# =====================================================
def encode_labels(label_str):
    vec = torch.zeros(num_classes)
    labels = label_str.split("|")
    for l in labels:
        if l in classes:
            vec[classes.index(l)] = 1.0
    return vec

# =====================================================
# IMAGE PATH
# =====================================================
def get_image_path(img_name):
    for i in range(1, 13):
        folder = f"images_{i:03d}"
        path = os.path.join(root_dir, folder, "images", img_name)
        if os.path.exists(path):
            return path
    return None

# =====================================================
# DATASET
# =====================================================
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
            return None, None

        image = Image.open(img_path).convert("RGB")
        label = encode_labels(label_str)

        if self.transform:
            image = self.transform(image)

        return image, label

# =====================================================
# COLLATE FUNCTION (Necessaria per gestire i None)
# =====================================================
def collate_fn(batch):
    # Filtra i sample corrotti o mancanti (None)
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0)
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return images, labels

# =====================================================
# IMAGE SIZE & TRANSFORMS (Allineato a 384 del Train)
# =====================================================
IMAGE_SIZE = 384

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDEVICE:", device)

# =====================================================
# LOAD VALIDATION DATA
# =====================================================
val_df = pd.read_csv(VAL_CSV)
val_dataset = NIHChestDataset(val_df, val_transform)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,  # Batch size ereditata dal tuo train
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn
)
print(f"Validation samples caricati: {len(val_df)}")

# =====================================================
# MODEL ARCHITECTURE
# =====================================================
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

# =====================================================
# LOAD WEIGHTS (Caricamento corretto dello state_dict)
# =====================================================
print(f"\nCaricamento pesi da: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Non ho trovato il file {MODEL_PATH}. Assicurati che sia nella stessa cartella.")

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
print("Modello caricato con successo ✔")

# =====================================================
# GENERATE PREDICTIONS
# =====================================================
model.eval()
all_probs = []
all_targets = []

print("\nGenerazione delle predizioni sul Validation Set...")
with torch.no_grad():
    for images, labels in tqdm(val_loader):
        if images.numel() == 0:  # Salta se il batch è vuoto a causa di immagini mancanti
            continue
            
        images = images.to(device)
        outputs = model(images)
        
        # Applichiamo il sigmoid per convertire i logits in probabilità (0-1)
        probs = torch.sigmoid(outputs)
        
        all_probs.append(probs.cpu())
        all_targets.append(labels)

# Concatenazione dei risultati
all_probs = torch.cat(all_probs).numpy()
all_targets = torch.cat(all_targets).numpy()

print(f"\nPredizioni completate. Shape Probs: {all_probs.shape} | Shape Targets: {all_targets.shape}")

# =====================================================
# THRESHOLD TUNING (Ottimizzazione F1-Score)
# =====================================================
print("\n========================")
print("RICERCA SOGLIE OTTIMALI (F1-SCORE)")
print("========================")

best_thresholds = {}

for i, cls in enumerate(classes):
    best_f1 = 0.0
    best_threshold = 0.5  # Default di partenza

    y_true = all_targets[:, i]
    y_prob = all_probs[:, i]

    # Ricerca della soglia a passi di 0.05
    for threshold in np.arange(0.05, 1.00, 0.05):
        y_pred = (y_prob >= threshold).astype(int)
        
        try:
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        except:
            pass

    best_thresholds[cls] = round(best_threshold, 2)
    print(f"{cls:25s} | Soglia Ottimale: {best_threshold:.2f} | Miglior F1-Score: {best_f1:.4f}")

# =====================================================
# SAVE THRESHOLDS IN JSON
# =====================================================
threshold_path = os.path.join(PYTHON_DIR, "optimized_thresholds.json")

with open(threshold_path, "w") as f:
    json.dump(best_thresholds, f, indent=4)

print("\n========================")
print("SOGLIE SALVATE CON SUCCESSO")
print("========================")
print(threshold_path)
print("\nPROCESSO COMPLETATO ✔")