import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from rad_dino import RadDino

# --- LOGGING ---
# Reindirizza tutto l'output su un file di log per non perdere traccia in caso di disconnessione
sys.stdout = open("training_progress.log", "a", buffering=1)

# --- 1. CONFIGURAZIONE ---
IMAGE_SIZE = 224
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
EPOCHS = 30
LR = 5e-5
PATIENCE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
train_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/rad_dino_best2.pth"

classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", 
           "Pleural_Thickening", "Hernia"]
num_classes = len(classes)

# --- 2. DATASET ---
class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.transform = transform
        # Mappatura immagini (fatta una sola volta all'inizializzazione)
        self.image_map = {img: os.path.join(root_dir, f"images_{i:03d}", "images", img) 
                          for i in range(1, 13) for img in os.listdir(os.path.join(root_dir, f"images_{i:03d}", "images"))}
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_map.get(row["Image Index"])
        img = Image.open(img_path).convert("RGB") if img_path else torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        if self.transform: img = self.transform(img)
        label_vec = torch.tensor([1.0 if l in str(row["Finding Labels"]).split("|") else 0.0 for l in classes])
        return img, label_vec

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- MODIFICA STABILITÀ: num_workers=0 e pin_memory=False ---
train_loader = DataLoader(NIHChestXrayDataset(train_csv, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=False)
val_loader = DataLoader(NIHChestXrayDataset(val_csv, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)

# --- 3. MODELLO ---
class RAD_DINO_Wrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = RadDino().model
        self.head = nn.Linear(768, num_classes)
    def forward(self, x): return self.head(self.backbone(x).last_hidden_state[:, 0, :])

model = RAD_DINO_Wrapper(num_classes).to(device)
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': LR * 0.1},
    {'params': model.head.parameters(), 'lr': LR}
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
scaler = torch.amp.GradScaler('cuda')

# --- 4. TRAINING LOOP ---
best_auc = 0.0
patience_counter = 0

print("Inizio training...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        with torch.amp.autocast('cuda'):
            loss = criterion(model(images), targets) / ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
            
        running_loss += loss.item() * ACCUMULATION_STEPS
    
    # Validazione
    model.eval()
    val_targets, val_outputs = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            val_outputs.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
            val_targets.append(targets.numpy())
    
    y_true, y_pred = np.vstack(val_targets), np.vstack(val_outputs)
    macro_auc = np.mean([roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(num_classes)])
    avg_loss = running_loss / len(train_loader)
    
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Macro-AUC: {macro_auc:.4f}")
    scheduler.step(macro_auc)
    
    if macro_auc > best_auc:
        best_auc = macro_auc
        patience_counter = 0
        torch.save(model.state_dict(), checkpoint_path)
        print("--> Modello migliorato e salvato.")
    else:
        patience_counter += 1
        print(f"--> Nessun miglioramento ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print("Early stopping attivato.")
            break