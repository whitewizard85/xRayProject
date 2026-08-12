import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import timm
import numpy as np
from sklearn.metrics import roc_auc_score

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archiveCheXpert"
train_csv = os.path.join(root_dir, "train.csv")
checkpoint_dir = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints"
log_file = "training_log.csv"
os.makedirs(checkpoint_dir, exist_ok=True)

TARGET_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", 
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", 
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

BATCH_SIZE = 8
ACCUMULATION_STEPS = 4 
LR_BACKBONE = 1e-6
LR_HEAD = 1e-4
EPOCHS = 10
PATIENCE_LIMIT = 5 

# ==========================================
# 2. DATASET (Logica confermata)
# ==========================================
class CheXpertDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path).fillna(0)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Pulisce il percorso come nello script ConvNeXt che ti funzionava
        path_in_csv = row["Path"].replace("CheXpert-v1.0-small/", "")
        img_path = os.path.join(self.root_dir, path_in_csv)
        
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            return torch.zeros(3, 600, 600), torch.zeros(len(TARGET_COLS))
            
        if self.transform: img = self.transform(img)
        labels = torch.tensor([max(0, float(row[c])) for c in TARGET_COLS], dtype=torch.float32)
        return img, labels

transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = CheXpertDataset(train_csv, root_dir, transform=transform)
val_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

# num_workers=0 per evitare freeze su VM
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ==========================================
# 3. MODELLO (Nome tf_efficientnet_b7)
# ==========================================
model = timm.create_model('tf_efficientnet_b7', pretrained=True, num_classes=len(TARGET_COLS))
model = model.to(device)

# Selezione pesi universale senza usare .features
optimizer = optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': LR_BACKBONE},
    {'params': [p for n, p in model.named_parameters() if 'classifier' in n], 'lr': LR_HEAD}
], weight_decay=1e-2)

criterion = nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()

# ==========================================
# 4. TRAINING LOOP
# ==========================================
best_auc = 0.0
best_val_loss = float('inf')
patience_counter = 0

print(f"Inizio training su {device}...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoca {epoch+1} [Train]")):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = criterion(model(images.to(device)), labels.to(device)) / ACCUMULATION_STEPS
        scaler.scale(loss).backward()
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
        train_loss += loss.item() * ACCUMULATION_STEPS

    model.eval()
    val_preds, val_targets, val_loss = [], [], 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoca {epoch+1} [Val]"):
            outputs = model(images.to(device))
            val_loss += criterion(outputs, labels.to(device)).item()
            val_preds.append(torch.sigmoid(outputs).cpu().numpy())
            val_targets.append(labels.numpy())
            
    macro_auc = roc_auc_score(np.vstack(val_targets), np.vstack(val_preds), average='macro')
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    log_data = pd.DataFrame([[epoch+1, avg_train_loss, avg_val_loss, macro_auc]], 
                            columns=['epoch', 'train_loss', 'val_loss', 'auc'])
    log_data.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
    
    print(f"\n>>> Epoca {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro AUC: {macro_auc:.4f}")
    
    # Controllo combinato: AUC deve migliorare E la Loss non deve esplodere
    if macro_auc >= best_auc and avg_val_loss <= best_val_loss:
        best_auc = macro_auc
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_b7_chexpert.pth"))
        print("🏆 Nuovo record AUC e Loss stabile, modello salvato!")
    else:
        patience_counter += 1
        print(f"📉 Nessun miglioramento combinato. Patience: {patience_counter}/{PATIENCE_LIMIT}")
        if patience_counter >= PATIENCE_LIMIT:
            print("\n⛔ Early stopping attivato!")
            break

print("\nTRAINING COMPLETED ✔")