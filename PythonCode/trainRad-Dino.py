import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from rad_dino import RadDino

# --- CONFIGURAZIONE ---
IMAGE_SIZE = 224
BATCH_SIZE = 4
ACCUMULATION_STEPS = 8
EPOCHS = 20
LR = 1e-5
PATIENCE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
train_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/rad_dino_final.pth"

classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", 
           "Pleural_Thickening", "Hernia"]
num_classes = len(classes)

# --- DATASET ---
class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.transform = transform
        self.image_map = {img: os.path.join(root_dir, f"images_{i:03d}", "images", img) 
                          for i in range(1, 13) for img in os.listdir(os.path.join(root_dir, f"images_{i:03d}", "images"))}
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_map.get(row["Image Index"])
        img = Image.open(img_path).convert("RGB") if img_path else torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        if self.transform: img = self.transform(img)
        label_vec = torch.zeros(num_classes)
        for l in str(row["Finding Labels"]).split("|"):
            if l in classes: label_vec[classes.index(l)] = 1.0
        return img, label_vec

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader = DataLoader(NIHChestXrayDataset(train_csv, transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(NIHChestXrayDataset(val_csv, transform), batch_size=BATCH_SIZE, shuffle=False)

# --- MODELLO RAD-DINO (FIXED) ---
class RAD_DINO_Wrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Carichiamo il modello ufficiale
        self.rad_dino = RadDino()
        # Accediamo direttamente al backbone di tipo Dinov2Model
        self.backbone = self.rad_dino.model
        self.head = nn.Linear(768, num_classes)
        
    def forward(self, x):
        # Passiamo il tensore direttamente al backbone (supporta GPU)
        outputs = self.backbone(x)
        # Estraiamo il token CLS (indice 0)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_token)

model = RAD_DINO_Wrapper(num_classes).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler('cuda')

# --- TRAINING LOOP ---
best_auc = 0.0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoca {epoch}")
    for i, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets) / ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
            
        running_loss += loss.item() * ACCUMULATION_STEPS
        pbar.set_postfix({'loss': running_loss / (i + 1)})

    # Validazione
    model.eval()
    val_targets, val_outputs = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = torch.sigmoid(model(images.to(device)))
            val_targets.append(targets.cpu().numpy())
            val_outputs.append(outputs.cpu().numpy())
    
    y_true, y_pred = np.vstack(val_targets), np.vstack(val_outputs)
    macro_auc = np.mean([roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(num_classes)])
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Macro-AUC: {macro_auc:.4f}")
    
    if macro_auc > best_auc:
        best_auc = macro_auc
        patience_counter = 0
        torch.save(model.state_dict(), checkpoint_path)
        print("--> Modello migliorato e salvato.")
    else:
        patience_counter += 1
        print(f"--> Nessun miglioramento. Pazienza: {patience_counter}/{PATIENCE}")
        
    if patience_counter >= PATIENCE:
        print("Early stopping attivato.")
        break