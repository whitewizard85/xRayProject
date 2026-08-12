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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# --- CONFIGURAZIONE ---
IMAGE_SIZE = 384
BATCH_SIZE = 4
ACCUMULATION_STEPS = 8
EPOCHS = 20
FREEZE_EPOCHS = 2
LR = 1e-4
PATIENCE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
train_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_multimodal_swin.pth"

classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", 
           "Pleural_Thickening", "Hernia"]
num_classes = len(classes)

# --- DATASET MULTIMODALE ---
class NIHMultimodalDataset(Dataset):
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
        
        # Etichette
        label_vec = torch.zeros(num_classes)
        for l in str(row["Finding Labels"]).split("|"):
            if l in classes: label_vec[classes.index(l)] = 1.0
            
        # Metadati (Encoding)
        age = float(row['Patient Age']) / 100.0
        gender = 0.0 if row['Patient Gender'] == 'M' else 1.0
        view = 0.0 if row['View Position'] == 'PA' else 1.0
        metadata = torch.tensor([age, gender, view], dtype=torch.float32)
        
        return img, label_vec, metadata

# --- MODELLO MULTIMODALE ---
class MultimodalSwin(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.swin = timm.create_model('swinv2_base_window12to24_192to384', pretrained=True, num_classes=0)
        self.feature_dim = self.swin.num_features 
        self.metadata_dim = 3 # Age, Gender, View
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + self.metadata_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x, metadata):
        features = self.swin(x)
        combined = torch.cat((features, metadata), dim=1)
        return self.classifier(combined)

# Inizializzazione
train_loader = DataLoader(NIHMultimodalDataset(train_csv, transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(NIHMultimodalDataset(val_csv, transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = MultimodalSwin(num_classes).to(device)
criterion = nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()

# --- TRAINING LOOP ---
best_auc, patience_counter = 0.0, 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    optimizer = optim.AdamW(model.parameters(), lr=LR) if epoch > FREEZE_EPOCHS else optim.Adam(model.classifier.parameters(), lr=1e-3)
    
    for i, (images, targets, metadata) in enumerate(tqdm(train_loader, desc=f"Epoca {epoch}")):
        images, targets, metadata = images.to(device), targets.to(device), metadata.to(device)
        
        with torch.cuda.amp.autocast():
            loss = criterion(model(images, metadata), targets) / ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            
        running_loss += loss.item() * ACCUMULATION_STEPS
    
    # Validazione
    model.eval()
    val_targets, val_outputs = [], []
    with torch.no_grad():
        for images, targets, metadata in tqdm(val_loader, desc="Validazione"):
            outputs = torch.sigmoid(model(images.to(device), metadata.to(device)))
            val_targets.append(targets.numpy()); val_outputs.append(outputs.cpu().numpy())
    
    macro_auc = np.mean([roc_auc_score(np.vstack(val_targets)[:, i], np.vstack(val_outputs)[:, i]) for i in range(num_classes)])
    print(f"Loss: {running_loss/len(train_loader):.4f} | AUC: {macro_auc:.4f}")
    
    if macro_auc > best_auc:
        best_auc, patience_counter = macro_auc, 0
        torch.save(model.state_dict(), checkpoint_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE: break