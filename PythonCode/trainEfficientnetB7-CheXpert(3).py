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
import torch.nn.functional as F

# =====================================================
# CONFIGURAZIONE ESTREMA
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
train_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
checkpoint_dir = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

IMAGE_SIZE = 1024
BATCH_SIZE = 2          # Batch ridotto al minimo per 1024px
ACCUMULATION_STEPS = 4  # Simula un batch size di 8
EPOCHS = 20
LR_BACKBONE = 5e-7
LR_HEAD = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", 
           "Fibrosis", "Pleural_Thickening", "Hernia"]
num_classes = len(classes)

# =====================================================
# 1. LOSS & DATASET
# =====================================================
class ExtremeFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, smoothing=0.15):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing
    def forward(self, inputs, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt)**self.gamma * bce).mean()

def get_image_path(img_name):
    for i in range(1, 13):
        path = os.path.join(root_dir, f"images_{i:03d}", "images", img_name)
        if os.path.exists(path): return path
    return None

class NIHDataset(Dataset):
    def __init__(self, csv, transform):
        self.df = pd.read_csv(csv)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = get_image_path(row["Image Index"])
        try: img = Image.open(path).convert("RGB")
        except: img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
        img = self.transform(img)
        label = torch.zeros(num_classes)
        for l in str(row["Finding Labels"]).split('|'):
            if l in classes: label[classes.index(l)] = 1.0
        return img, label

# =====================================================
# 2. MODELLO E OTTIMIZZATORE
# =====================================================
model = timm.create_model('tf_efficientnet_b7', pretrained=True, num_classes=num_classes)
model.set_grad_checkpointing(True)
model = model.to(device)

optimizer = optim.AdamW([
    {'params': model.conv_stem.parameters(), 'lr': LR_BACKBONE},
    {'params': model.blocks.parameters(), 'lr': LR_BACKBONE},
    {'params': model.classifier.parameters(), 'lr': LR_HEAD}
])

train_loader = DataLoader(NIHDataset(train_csv, transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

val_loader = DataLoader(NIHDataset(val_csv, transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])),
    batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# =====================================================
# 3. LOOP ESTREMO
# =====================================================
criterion = ExtremeFocalLoss()
best_auc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    
    for i, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoca {epoch}")):
        images, targets = images.to(device), targets.to(device)
        with torch.amp.autocast('cuda'):
            loss = criterion(model(images), targets) / ACCUMULATION_STEPS
        loss.backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item() * ACCUMULATION_STEPS

    # Validazione
    model.eval()
    val_out, val_targ = [], []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validazione"):
            preds = torch.sigmoid(model(images.to(device)))
            val_out.append(preds.cpu().numpy())
            val_targ.append(targets.numpy())
            
    auc = np.mean([roc_auc_score(np.vstack(val_targ)[:, i], np.vstack(val_out)[:, i]) for i in range(num_classes)])
    print(f"Loss: {running_loss/len(train_loader):.4f} | AUC: {auc:.4f}")
    
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_nih_extreme.pth"))
        print("🏆 Record aggiornato!")