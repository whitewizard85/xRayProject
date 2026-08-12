import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Necessario per l'estrazione delle feature

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torchxrayvision as xrv

# =====================================================
# ASYMMETRIC LOSS (ASL) 
# =====================================================
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        xs = torch.sigmoid(x)
        xs_pos = xs
        xs_neg = 1.0 - xs

        if self.clip and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        loss_pos = y * torch.log(xs_pos.clamp(min=self.eps)) * ((1.0 - xs_pos) ** self.gamma_pos)
        loss_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps)) * ((1.0 - xs_neg) ** self.gamma_neg)
        
        loss = loss_pos + loss_neg
        return -loss.mean()

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDEVICE DI ADDESTRAMENTO:", device)

# =====================================================
# PATHS
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
train_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv"
val_csv   = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"

# =====================================================
# CLASSES
# =====================================================
classes = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax",
    "Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia"
]
num_classes = len(classes)

def encode_labels(label_str):
    vec = torch.zeros(num_classes)
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

# =====================================================
# DATASET ADATTATO PER TORCHXRAYVISION (1 Canale + Scala [-1024, 1024])
# =====================================================
class NIHChestDatasetXRV(Dataset):
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
        if img_path is None:
            return None, None

        # XRV richiede tassativamente la scala di grigi (L)
        image = Image.open(img_path).convert("L")
        
        # Trasformiamo in array numpy prima di applicare i filtri custom di XRV
        img_np = np.array(image)
        
        # Conversione fondamentale richiesta dai pesi XRV: scala [-1024, 1024]
        img_np = xrv.datasets.normalize(img_np, maxval=255)
        
        # Riportiamo in formato PIL Image per permettere i transform di torchvision
        image = Image.fromarray(img_np)

        label = encode_labels(label_str)

        if self.transform:
            image = self.transform(image)

        return image, label

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0)
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return images, labels

# =====================================================
# TRANSFORMS ADATTATE (Niente Normalize RGB di ImageNet!)
# =====================================================
IMAGE_SIZE = 512

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor() # Restituisce 1 canale nativo
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# =====================================================
# LOAD DATA
# =====================================================
train_df = pd.read_csv(train_csv)
val_df   = pd.read_csv(val_csv)

train_dataset = NIHChestDatasetXRV(train_df, train_transform)
val_dataset   = NIHChestDatasetXRV(val_df, val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)

# =====================================================
# ARCHITETTURA BIOMEDICALE (FEATURES EXTRACTION PURA)
# =====================================================
# 1. Carichiamo l'architettura base pre-addestrata sui dataset polmonari mondiali
base_model = xrv.models.DenseNet(weights="densenet121-res224-all")

# 2. Creiamo una classe contenitore per fare l'override del flusso di inferenza originale
class XRVFeatureExtractor(nn.Module):
    def __init__(self, xrv_model, num_classes):
        super(XRVFeatureExtractor, self).__init__()
        self.features = xrv_model.features           # Tronco convoluzionale pre-addestrato
        self.classifier = nn.Linear(1024, num_classes) # Testa lineare pulita per NIH (14 classi)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

model = XRVFeatureExtractor(base_model, num_classes)
model = model.to(device)

print("\nDenseNet121 TorchXRayVision con Estrattore di Feature Isolato Caricata ✔")

# =====================================================
# CONFIGURAZIONE STRATEGICA
# =====================================================
criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

epochs = 30
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-4, steps_per_epoch=len(train_loader), epochs=epochs
)

scaler = torch.cuda.amp.GradScaler()

patience = 10
no_improve = 0
best_auc = 0

# =====================================================
# TRAINING LOOP
# =====================================================
print("\nSTART TRAINING (Variante v4: 512px + ASL + TorchXRayVision Weights)")

for epoch in range(epochs):
    # TRAIN
    model.train()
    train_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        if images.numel() == 0: continue
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # VALIDATION
    model.eval()
    val_loss = 0
    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
            if images.numel() == 0: continue
            images = images.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item()

            # Detach fondamentale per azzerare i gradienti di inferenza e ripulire la memoria
            preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            targets.append(labels.cpu().numpy())

    preds = np.vstack(preds)
    targets = np.vstack(targets)
    auc = roc_auc_score(targets, preds, average="macro")

    print(f"\n--- Risultati Epoca {epoch+1} ---")
    print("Train Loss:", round(train_loss, 4))
    print("Val Loss:", round(val_loss/len(val_loader), 4))
    print("ROC-AUC Validazione:", round(auc, 4))

    # SAVE BEST
    os.makedirs("checkpoints", exist_ok=True)
    if auc > best_auc:
        best_auc = auc
        no_improve = 0
        torch.save(model.state_dict(), "checkpoints/best_densenet121_v4_xrv.pth")
        print("✔ MIGLIOR MODELLO SALVATO (checkpoints/best_densenet121_v4_xrv.pth)")
    else:
        no_improve += 1
        print(f"Nessun miglioramento: {no_improve}/{patience}")

    if no_improve >= patience:
        print("\n⛔ EARLY STOPPING TRIGGERED")
        break

print("\nTRAINING BIOMEDICALE COMPLETATO!")