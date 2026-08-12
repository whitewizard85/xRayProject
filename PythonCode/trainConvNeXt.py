import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import timm

# =====================================================
# CONFIGURAZIONE PATH E PARAMETRI
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
train_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
checkpoint_dir = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Iperparametri
BATCH_SIZE = 16  # Bilanciamento sicuro per ConvNeXt-Base su una singola GPU
EPOCHS = 10
LR = 3e-5        # Learning rate conservativo, ideale per il fine-tuning di ConvNeXt
IMAGE_SIZE = 384 # Risoluzione nativa ottimale per ConvNeXt-Base in22k

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Dispositivo di addestramento: {device}")

# =====================================================
# ASYMMETRIC LOSS (ASL) INDIPENDENTE
# =====================================================
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, xs, ys):
        xs_sig = torch.sigmoid(xs)
        xs_pos = xs_sig
        xs_neg = 1.0 - xs_sig

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        loss_pos = ys * torch.log(xs_pos.clamp(min=self.eps)) * ((1 - xs_pos) ** self.gamma_pos)
        loss_neg = (1 - ys) * torch.log(xs_neg.clamp(min=self.eps)) * ((1 - xs_neg) ** self.gamma_neg)
        loss = -1 * (loss_pos + loss_neg)
        return loss.mean()

# =====================================================
# DATASET E TRASFORMAZIONI
# =====================================================
def get_image_path(img_name):
    for i in range(1, 13):
        folder = f"images_{i:03d}"
        path = os.path.join(root_dir, folder, "images", img_name)
        if os.path.exists(path): return path
    return None

class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.transform = transform
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = get_image_path(img_name)
        
        # Gestione fall-back se l'immagine è corrotta o mancante
        if img_path is None or not os.path.exists(img_path):
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), torch.zeros(num_classes)
            
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        # Encoding multi-label
        label_vec = torch.zeros(num_classes)
        labels = str(row["Finding Labels"]).split("|")
        for l in labels:
            if l in classes:
                label_vec[classes.index(l)] = 1.0
                
        return img, label_vec

# Trasformazioni (Data Augmentation leggera per il Train, solo Resize/Normalize per Val)
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = NIHChestXrayDataset(train_csv, transform=train_transform)
val_dataset = NIHChestXrayDataset(val_csv, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# =====================================================
# INIZIALIZZAZIONE CONVNEXT-BASE (IMAGENET-22K)
# =====================================================
print("[INIT] Caricamento ConvNeXt-Base con pesi pre-addestrati ImageNet-22k...")
# timm scaricherà automaticamente il checkpoint ufficiale di Meta AI 'convnext_base.fb_in22k'
model = timm.create_model('convnext_base.fb_in22k', pretrained=True, num_classes=num_classes)
model = model.to(device)

criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# =====================================================
# LOOP DI ADDESTRAMENTO E VALIDAZIONE
# =====================================================
best_macro_auc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    print(f"\n--- Epoca {epoch}/{EPOCHS} ---")
    
    for images, targets in tqdm(train_loader, desc="Training"):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
    scheduler.step()
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Train Loss: {epoch_loss:.4f}")
    
    # Validazione
    model.eval()
    val_targets = []
    val_outputs = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validazione"):
            images = images.to(device)
            outputs = torch.sigmoid(model(images))
            
            val_targets.append(targets.numpy())
            val_outputs.append(outputs.cpu().numpy())
            
    val_targets = np.vstack(val_targets)
    val_outputs = np.vstack(val_outputs)
    
    # Calcolo ROC-AUC per classe
    auc_scores = []
    for i in range(num_classes):
        try:
            auc = roc_auc_score(val_targets[:, i], val_outputs[:, i])
            auc_scores.append(auc)
        except ValueError:
            auc_scores.append(0.5) # Fallback se mancano positivi nel batch/split
            
    macro_auc = np.mean(auc_scores)
    print(f"Validation Macro ROC-AUC: {macro_auc:.4f}")
    
    # Salvataggio del modello migliore
    if macro_auc > best_macro_auc:
        best_macro_auc = macro_auc
        checkpoint_path = os.path.join(checkpoint_dir, "best_convnext_base_22k.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"🏆 Nuovo record! Checkpoint salvato in: {checkpoint_path}")

print(f"\n[FINISH] Addestramento concluso! Miglior Validation Macro AUC: {best_macro_auc:.4f} 🚀")