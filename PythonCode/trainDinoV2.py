import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# --- CONFIGURAZIONE ---
IMAGE_SIZE = 224
BATCH_SIZE = 4
ACCUMULATION_STEPS = 8
EPOCHS = 20
PATIENCE = 8
LR = 5e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
train_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_model_v2.pth"

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

train_loader = DataLoader(NIHChestXrayDataset(train_csv, transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(NIHChestXrayDataset(val_csv, transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- MODELLO ---
print(f"Caricamento DINOv2 per risoluzione {IMAGE_SIZE}...")
model = timm.create_model(
    'vit_base_patch14_dinov2.lvd142m', 
    pretrained=True, 
    num_classes=num_classes,
    img_size=IMAGE_SIZE
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler('cuda')

# --- TRAINING LOOP ---
best_auc, patience_counter = 0.0, 0
for epoch in range(1, EPOCHS + 1):
    
    # Gestione Gradiente: blocca il backbone ma lascia libero il classificatore (head)
    if epoch <= 2:
        model.requires_grad_(False)
        model.head.requires_grad_(True)
    else:
        model.requires_grad_(True)

    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    for i, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoca {epoch}")):
        images, targets = images.to(device), targets.to(device)
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets) / ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        running_loss += loss.item() * ACCUMULATION_STEPS

    # Validazione
    model.eval()
    val_targets, val_outputs = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = torch.sigmoid(model(images.to(device)))
            val_targets.append(targets.numpy())
            val_outputs.append(outputs.cpu().numpy())
    
    y_true, y_pred = np.vstack(val_targets), np.vstack(val_outputs)
    macro_auc = np.mean([roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(num_classes)])
    
    print(f"Epoch {epoch} | Loss: {running_loss/len(train_loader):.4f} | Macro-AUC: {macro_auc:.4f}")
    
    if macro_auc > best_auc:
        best_auc = macro_auc
        torch.save(model.state_dict(), checkpoint_path)
        print("--> Modello salvato.")