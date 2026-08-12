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
# 1. PATHS E CONFIGURAZIONE
# =====================================================
PYTHON_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode"
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
VAL_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"

# Proviamo la cartella principale (dove ti si salvano i file se lanci il train da lì)
MODEL_PATH = "/home/gpuvm/Desktop/Luca Migliaccio/best_efficientnet_b7.pth"

# SE DA ERRORE ANCHE QUI, COMMENTA LA RIGA SOPRA E SCOMMENTA QUESTA SOTTO (cartella checkpoints):
# MODEL_PATH = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/checkpoints/best_efficientnet_b7.pth"

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
NUM_CLASSES = len(classes)
BATCH_SIZE = 16
IMAGE_SIZE = 600  # Allineato al tuo train (B7 @ 600x600)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nEsecuzione su: {device}")

# =====================================================
# 2. DATASET (Esatta copia della logica del tuo train)
# =====================================================
def encode_labels(label_str):
    vec = torch.zeros(NUM_CLASSES)
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
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = encode_labels(label_str)
        return image, label

# =====================================================
# 3. DATA LOADER (num_workers=0 per massima stabilità)
# =====================================================
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize
])

val_df = pd.read_csv(VAL_CSV)
val_dataset = NIHChestDataset(val_df, val_transform)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=0,  
    pin_memory=True
)
print(f"Campioni di validazione caricati: {len(val_df)}")

# =====================================================
# 4. CARICAMENTO MODELLO (EfficientNet-B7)
# =====================================================
print("\nInizializzazione EfficientNet-B7...")
model = models.efficientnet_b7(weights=None)  # Struttura vuota
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

print(f"Caricamento pesi da: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"\n[ERRORE] Il file {MODEL_PATH} non esiste!")
    print("Controlla se per caso è dentro la cartella checkpoints.")
    raise FileNotFoundError()

# Caricamento dello state_dict
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print("Modello caricato con successo ✔")

# =====================================================
# 5. INFERENZA CON AMP (Allineato al train)
# =====================================================
val_preds = []
val_targets = []

print("\nGenerazione delle predizioni sul Validation Set (16565 campioni)...")
with torch.no_grad():
    for images, labels in tqdm(val_loader):
        images = images.to(device)
        
        # Usiamo l'autocast a 16-bit come nel tuo train per evitare Out of Memory
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            
        outputs = torch.sigmoid(outputs)  # Conversione in probabilità
        val_preds.append(outputs.cpu())
        val_targets.append(labels.cpu())

all_preds = torch.cat(val_preds).numpy()
all_targets = torch.cat(val_targets).numpy()

# =====================================================
# 6. RICERCA SOGLIE OTTIMALI
# =====================================================
print("\n==========================================")
print("RICERCA SOGLIE OTTIMALI (F1-SCORE)")
print("==========================================")

best_thresholds = {}

for i, cls in enumerate(classes):
    best_f1 = 0.0
    best_threshold = 0.5

    y_true = all_targets[:, i]
    y_prob = all_preds[:, i]

    # Ricerca granulare della soglia passo 0.01
    for threshold in np.arange(0.01, 1.00, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    best_thresholds[cls] = round(best_threshold, 2)
    print(f"{cls:25s} | Soglia Ottimale: {best_threshold:.2f} | Miglior F1-Score: {best_f1:.4f}")

# =====================================================
# 7. SALVATAGGIO JSON
# =====================================================
output_json_path = os.path.join(PYTHON_DIR, "optimized_thresholds_EfficientNetB7.json")
with open(output_json_path, "w") as f:
    json.dump(best_thresholds, f, indent=4)

print("\n==========================================")
print("SOGLIE SALVATE CON SUCCESSO ✔")
print(output_json_path)
print("==========================================")