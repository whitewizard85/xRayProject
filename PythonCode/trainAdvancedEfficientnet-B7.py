import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# =====================================================
# 1. CLASSE ASYMMETRIC LOSS (ASL) DEFINITIVA
# =====================================================
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        # Calcolo delle probabilità tramite sigmoide
        xs_p = torch.sigmoid(x)
        xs_n = 1 - xs_p

        # Asymmetric Clipping per i campioni negativi facili
        if self.clip and self.clip > 0:
            xs_n = (xs_n + self.clip).clamp(max=1)

        # Basic BCE loss
        loss_pos = y * torch.log(xs_p.clamp(min=self.eps))
        loss_neg = (1 - y) * torch.log(xs_n.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.gamma_pos > 0:
            loss_pos *= (1 - xs_p) ** self.gamma_pos
        if self.gamma_neg > 0:
            loss_neg *= (1 - xs_n) ** self.gamma_neg

        loss = - (loss_pos + loss_neg)
        return loss.mean()

# =====================================================
# 2. CONFIGURAZIONE PATHS ED IPERPARAMETRI
# =====================================================
BASE_DIR = "/home/gpuvm/Desktop/Luca Migliaccio"
PYTHON_DIR = os.path.join(BASE_DIR, "PythonCode")
root_dir = os.path.join(BASE_DIR, "archive")

TRAIN_CSV = os.path.join(PYTHON_DIR, "train_split.csv")
VAL_CSV = os.path.join(PYTHON_DIR, "val_split.csv")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "best_efficientnet_b7_asl.pth")

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
NUM_CLASSES = len(classes)
BATCH_SIZE = 16       # Ottimale per B7 @ 600x600 su GPU da 32GB
IMAGE_SIZE = 600
LEARNING_RATE = 1e-5  # Basso per preservare le feature estratte
EPOCHS = 15           # Più epoche grazie al controllo dell'overfitting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[INFO] Configurazione avviata su device: {device}")

# =====================================================
# 3. UTILS ED INDICIZZAZIONE DISCO RAPIDA
# =====================================================
def encode_labels(label_str):
    vec = torch.zeros(NUM_CLASSES)
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

class NIHChestDatasetAdvanced(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        
        print(" -> Caching preventivo dei percorsi su disco...")
        self.path_cache = {}
        for idx in tqdm(range(len(self.df)), desc="Indicizzazione"):
            img_name = self.df.iloc[idx]["Image Index"]
            if img_name not in self.path_cache:
                path = get_image_path(img_name)
                if path:
                    self.path_cache[img_name] = path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        label_str = row["Finding Labels"]
        
        img_path = self.path_cache.get(img_name)
        if img_path is None:
            img_path = get_image_path(img_name) # Fallback alternativo
            
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        label = encode_labels(label_str)
        return image, label

# =====================================================
# 4. PIPELINE DI DATA AUGMENTATION STRUTTURATA
# =====================================================
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15), # Previene la sintonizzazione geometrica fissa
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Combatte i bias di contrasto dei macchinari
    transforms.ToTensor(),
    normalize
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize
])

print("\n[CARICAMENTO DATASET] Inizializzazione in corso...")
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

train_dataset = NIHChestDatasetAdvanced(train_df, train_transform)
val_dataset = NIHChestDatasetAdvanced(val_df, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# =====================================================
# 5. RETE E COMPONENTI DI OTTIMIZZAZIONE
# =====================================================
print("\n[MODELLO] Configurazione di EfficientNet-B7 con pesi ImageNet...")
model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
model = model.to(device)

criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_schescheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
scaler = torch.amp.GradScaler()

# =====================================================
# 6. TRAINING E VALIDATION LOOP AD ALTA EFFICIENZA
# =====================================================
best_macro_auc = 0.0

print("\n[START] Inizio ciclo di addestramento avanzato...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item() * images.size(0)
        
    epoch_train_loss = train_loss / len(train_loader.dataset)
    
    # Validation phase
    model.eval()
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            images = images.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            val_preds.append(probs.cpu().numpy())
            val_targets.append(labels.numpy())
            
    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)
    
    # Calcolo Macro ROC-AUC di epoca
    auc_scores = []
    for i in range(NUM_CLASSES):
        try:
            auc_scores.append(roc_auc_score(val_targets[:, i], val_preds[:, i]))
        except ValueError:
            pass
            
    epoch_macro_auc = np.mean(auc_scores)
    print(f" -> Loss Train: {epoch_train_loss:.4f} | Macro ROC-AUC Val: {epoch_macro_auc:.4f}")
    
    # Aggiornamento Scheduler e salvataggio dei pesi migliori
    scheduler.step(epoch_macro_auc)
    
    if epoch_macro_auc > best_macro_auc:
        best_macro_auc = epoch_macro_auc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f" ✔ Nuovo record trovato! Modello salvato in: {BEST_MODEL_PATH}")

print(f"\n[COMPLETATO] Addestramento terminato. Miglior Macro ROC-AUC registrato: {best_macro_auc:.4f}")