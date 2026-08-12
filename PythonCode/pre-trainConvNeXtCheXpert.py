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
from torch.optim.lr_scheduler import CosineAnnealingLR

# =====================================================
# CONFIGURAZIONE
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archiveCheXpert"
train_csv_path = os.path.join(root_dir, "train.csv")
val_csv_path = os.path.join(root_dir, "valid.csv")
checkpoint_dir = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
IMAGE_SIZE = 384
PATIENCE = 8  # Parametro modificabile per l'Early Stopping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_classes = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion", 
                  "Pneumonia", "Pneumothorax", "No Finding", "Enlarged Cardiomediastinum", 
                  "Lung Lesion", "Lung Opacity", "Pleural Other", "Fracture", "Support Devices"]
num_classes = len(target_classes)

# =====================================================
# DATASET
# =====================================================
class CheXpertDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Path"].replace("CheXpert-v1.0-small/", ""))
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), torch.zeros(num_classes)
        if self.transform: img = self.transform(img)
        labels = torch.tensor([1.0 if row[c] == 1.0 else 0.0 for c in target_classes], dtype=torch.float32)
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

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

# =====================================================
# MODELLO E TRAINING
# =====================================================
model = timm.create_model('convnext_base.fb_in22k', pretrained=True, num_classes=num_classes).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.BCEWithLogitsLoss()

best_macro_auc = 0.0
counter = 0

for epoch in range(1, EPOCHS + 1):
    # --- Training ---
    model.train()
    running_train_loss = 0.0
    for images, targets in tqdm(train_loader, desc=f"Epoca {epoch} - Train"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * images.size(0)
    
    avg_train_loss = running_train_loss / len(train_ds)
    
    # --- Validazione ---
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
    macro_auc = roc_auc_score(np.vstack(all_targets), np.vstack(all_probs), average="macro")
    
    print(f"\n--- Epoca {epoch} Report ---")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Macro AUC: {macro_auc:.4f}\n")
    
    # --- Early Stopping ---
    if macro_auc > best_macro_auc:
        best_macro_auc = macro_auc
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_convnext_chexpert.pth"))
        counter = 0
    else:
        counter += 1
        print(f"AUC non migliorata. Counter: {counter}/{PATIENCE}")
        if counter >= PATIENCE:
            print("Early stopping raggiunto.")
            break
            
    scheduler.step()

print(f"\n[FINISH] Addestramento concluso! Miglior AUC: {best_macro_auc:.4f}")