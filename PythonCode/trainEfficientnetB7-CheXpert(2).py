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
# CONFIGURAZIONE
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
train_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
checkpoint_dir = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints"
pretrained_path = os.path.join(checkpoint_dir, "best_nih_model.pth")
os.makedirs(checkpoint_dir, exist_ok=True)

BATCH_SIZE = 8
IMAGE_SIZE = 600
EPOCHS = 20
PATIENCE = 8
LR = 1e-6 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", 
           "Fibrosis", "Pleural_Thickening", "Hernia"]
num_classes = len(classes)

# =====================================================
# 1. FOCAL LOSS & PESATURA DINAMICA
# =====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        return focal_loss.mean()

# Calcolo pesi basato su Finding Labels (formato |)
df_train = pd.read_csv(train_csv)
label_counts = {c: 0 for c in classes}
for labels in df_train["Finding Labels"].astype(str):
    for label in labels.split('|'):
        if label in label_counts:
            label_counts[label] += 1

pos_counts = torch.tensor([label_counts[c] for c in classes], dtype=torch.float)
neg_counts = len(df_train) - pos_counts
pos_weight = neg_counts / (pos_counts + 1.0)
criterion = FocalLoss(pos_weight=pos_weight.to(device))

# =====================================================
# 2. DATASET ROBUSTO
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
        path = get_image_path(img_name)
        try:
            img = Image.open(path).convert("RGB")
        except:
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
        if self.transform: img = self.transform(img)
        
        label_vec = torch.zeros(num_classes)
        labels = str(row["Finding Labels"]).split('|')
        for l in labels:
            if l in classes: label_vec[classes.index(l)] = 1.0
        return img, label_vec

# =====================================================
# 3. LOOP DI ADDESTRAMENTO
# =====================================================
model = timm.create_model('tf_efficientnet_b7', pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(pretrained_path, map_location=device))
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)

train_loader = DataLoader(NIHChestXrayDataset(train_csv, transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])), batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
val_loader = DataLoader(NIHChestXrayDataset(val_csv, transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])), batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

best_macro_auc = 0.0
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(train_loader, desc=f"Epoca {epoch}"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = criterion(model(images), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    
    model.eval()
    val_targets, val_outputs = [], []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validazione TTA"):
            img = images.to(device)
            # TTA: Originale, Flip, Rotata
            flip = torch.flip(img, [3])
            rot = transforms.functional.rotate(img, 10)
            preds = torch.sigmoid(model(torch.cat([img, flip, rot]))).mean(dim=0)
            val_outputs.append(preds.cpu().numpy())
            val_targets.append(targets.numpy())

    macro_auc = np.mean([roc_auc_score(np.vstack(val_targets)[:, i], np.vstack(val_outputs)[:, i]) for i in range(num_classes)])
    print(f"Train Loss: {running_loss/len(train_loader):.4f} | Val Macro AUC: {macro_auc:.4f}")
    
    if macro_auc > best_macro_auc:
        best_macro_auc = macro_auc
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_nih_final.pth"))
        print("🏆 Nuovo record salvato.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("\n[STOP] Early stopping attivato.")
            break