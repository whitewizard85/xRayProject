import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import timm
import pandas as pd
import numpy as np

# =====================================================
# CONFIGURAZIONE PATH E PARAMETRI
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
train_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
checkpoint_dir = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints"
pretrained_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/convnext_chexpert_pretrained.pth"
os.makedirs(checkpoint_dir, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 15
PATIENCE = 7
LR = 1e-5
IMAGE_SIZE = 384

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        if img_path is None or not os.path.exists(img_path):
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), torch.zeros(num_classes)
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        label_vec = torch.zeros(num_classes)
        labels = str(row["Finding Labels"]).split("|")
        for l in labels:
            if l in classes: label_vec[classes.index(l)] = 1.0
        return img, label_vec

train_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
val_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_loader = DataLoader(NIHChestXrayDataset(train_csv, transform=train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(NIHChestXrayDataset(val_csv, transform=val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# =====================================================
# INIZIALIZZAZIONE MODELLO (CARICAMENTO CHIRURGICO)
# =====================================================
print("[INIT] Inizializzazione modello...")
model = timm.create_model('convnext_base', pretrained=False, num_classes=num_classes)
checkpoint = torch.load(pretrained_path, map_location=device)

# Caricamento selettivo: carichiamo solo i pesi del backbone, non della testa
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and "head" not in k}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

# =====================================================
# LOOP DI ADDESTRAMENTO
# =====================================================
best_macro_auc = 0.0
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    print(f"\n--- Epoca {epoch}/{EPOCHS} ---")
    for images, targets in tqdm(train_loader, desc="Training"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    # Validazione
    model.eval()
    val_targets, val_outputs = [], []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validazione"):
            images = images.to(device)
            val_targets.append(targets.numpy())
            val_outputs.append(torch.sigmoid(model(images)).cpu().numpy())
    
    val_targets, val_outputs = np.vstack(val_targets), np.vstack(val_outputs)
    auc_scores = [roc_auc_score(val_targets[:, i], val_outputs[:, i]) for i in range(num_classes)]
    macro_auc = np.mean(auc_scores)
    
    print(f"Train Loss: {running_loss/len(train_loader.dataset):.4f} | Val Macro AUC: {macro_auc:.4f}")
    
    scheduler.step(macro_auc)
    if macro_auc > best_macro_auc:
        best_macro_auc, epochs_no_improve = macro_auc, 0
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_nih_model.pth"))
        print("🏆 Nuovo record salvato.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("\n[STOP] Early stopping attivato.")
            break