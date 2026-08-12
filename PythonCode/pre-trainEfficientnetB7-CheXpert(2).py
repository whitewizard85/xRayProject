import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# =====================================================
# 1. CONFIGURAZIONE
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archiveCheXpert"
train_csv_path = os.path.join(root_dir, "train.csv")
val_csv_path = os.path.join(root_dir, "valid.csv")
checkpoint_dir = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "best_convnext_chexpert.pth")

TARGET_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", 
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", 
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

BATCH_SIZE = 16
IMAGE_SIZE = 384
EPOCHS = 20
LR = 1e-4
PATIENCE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# 2. DATASET (GESTIONE INCERTI -1 E FILLNA)
# =====================================================
class CheXpertDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file).fillna(0)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Path"].replace("CheXpert-v1.0-small/", ""))
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), torch.zeros(len(TARGET_COLS))
        if self.transform: img = self.transform(img)
        
        # Mappatura: -1 diventa 0.0 per coerenza con il training precedente
        labels = torch.tensor([max(0.0, float(row[c])) for c in TARGET_COLS], dtype=torch.float32)
        return img, labels

train_ds = CheXpertDataset(train_csv_path, root_dir, transform=transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))
val_ds = CheXpertDataset(val_csv_path, root_dir, transform=transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))

# num_workers=0 per evitare saturazione memoria VM
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# =====================================================
# 3. MODELLO
# =====================================================
model = timm.create_model('convnext_base.fb_in22k', pretrained=True, num_classes=len(TARGET_COLS)).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

best_macro_auc = 0.0
counter = 0

# =====================================================
# 4. TRAINING LOOP
# =====================================================
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_train_loss = 0.0
    
    # Training
    for images, targets in tqdm(train_loader, desc=f"Epoca {epoch} - Train"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * images.size(0)
    
    avg_train_loss = running_train_loss / len(train_ds)
    
    # Validazione
    model.eval()
    running_val_loss = 0.0
    all_targets, all_probs = [], []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Epoca {epoch} - Val"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item() * images.size(0)
            all_probs.append(torch.sigmoid(outputs).cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    avg_val_loss = running_val_loss / len(val_ds)
    
    # Calcolo AUC robusto
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    valid_cols = [i for i in range(len(TARGET_COLS)) if len(np.unique(all_targets[:, i])) == 2]
    
    macro_auc = roc_auc_score(all_targets[:, valid_cols], all_probs[:, valid_cols], average="macro") if len(valid_cols) > 0 else 0.0
    
    # Report a schermo
    print(f"\n{'='*40}")
    print(f"EPOCA {epoch} COMPLETATA")
    print(f"Train Loss : {avg_train_loss:.4f}")
    print(f"Val Loss   : {avg_val_loss:.4f}")
    print(f"Macro AUC  : {macro_auc:.4f}")
    print(f"{'='*40}\n")
    
    # Early Stopping
    if macro_auc > best_macro_auc:
        best_macro_auc = macro_auc
        torch.save(model.state_dict(), checkpoint_path)
        counter = 0
    else:
        counter += 1
        if counter >= PATIENCE: 
            print("Early stopping raggiunto.")
            break

print(f"\n[FINISH] Addestramento concluso! Miglior AUC: {best_macro_auc:.4f}")