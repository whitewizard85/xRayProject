import os
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

# --- CONFIGURAZIONE ---
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
train_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
checkpoint_dir = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 20
PATIENCE = 8
LR = 3e-5
IMAGE_SIZE = 384
classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET CON DEBIASING ---
def get_image_path(img_name):
    for i in range(1, 13):
        path = os.path.join(root_dir, f"images_{i:03d}", "images", img_name)
        if os.path.exists(path): return path
    return None

class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.transform = transform
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = get_image_path(row["Image Index"])
        if img_path is None: return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), torch.zeros(len(classes))
        
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        
        label_vec = torch.zeros(len(classes))
        for l in str(row["Finding Labels"]).split("|"):
            if l in classes: label_vec[classes.index(l)] = 1.0
        return img, label_vec

# Trasformazioni con CenterCrop per rimuovere i bias dei bordi
transform_pipeline = transforms.Compose([
    transforms.Resize((420, 420)),
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader = DataLoader(NIHChestXrayDataset(train_csv, transform=transform_pipeline), batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
val_loader = DataLoader(NIHChestXrayDataset(val_csv, transform=transform_pipeline), batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

# --- MODELLO ---
print("[INIT] Caricamento ConvNeXt-Base (Pesi ImageNet-22k)...")
model = timm.create_model('convnext_base.fb_in22k', pretrained=True, num_classes=len(classes)).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# --- TRAINING LOOP ---
best_auc = 0.0
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(train_loader, desc=f"Epoca {epoch} [Train]"):
        optimizer.zero_grad()
        outputs = model(images.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    model.eval()
    val_outs, val_targs = [], []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Epoca {epoch} [Val]"):
            val_outs.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
            val_targs.append(targets.numpy())
            
    val_outs = np.vstack(val_outs)
    val_targs = np.vstack(val_targs)
    auc_scores = [roc_auc_score(val_targs[:, i], val_outs[:, i]) for i in range(len(classes))]
    macro_auc = np.mean(auc_scores)
    
    print(f"--> Epoca {epoch}: Loss={epoch_loss:.4f} | Macro-AUC={macro_auc:.4f}")
    
    if macro_auc > best_auc:
        best_auc = macro_auc
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_debiased_convnext.pth"))
        print("🏆 Miglior AUC raggiunto: Checkpoint salvato.")
    else:
        epochs_no_improve += 1
        print(f"⚠️ Nessun miglioramento (Pazienza: {epochs_no_improve}/{PATIENCE})")
        
    if epochs_no_improve >= PATIENCE:
        print("\n[STOP] Early stopping attivato.")
        break