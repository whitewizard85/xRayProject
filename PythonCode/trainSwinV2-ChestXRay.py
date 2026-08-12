import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForImageClassification, get_cosine_schedule_with_warmup

# --- CONFIGURAZIONE ---
IMAGE_SIZE = 224 
BATCH_SIZE = 4
ACCUMULATION_STEPS = 8  
EPOCHS = 20
FREEZE_EPOCHS = 2       
LR = 5e-5               
PATIENCE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
train_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_swin_biomedical.pth"

classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", 
           "Pleural_Thickening", "Hernia"]
num_classes = len(classes)

# --- DATASET ---
class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.transform = transform
        self.image_map = {}
        for i in range(1, 13):
            folder = os.path.join(root_dir, f"images_{i:03d}", "images")
            if os.path.exists(folder):
                for img_name in os.listdir(folder):
                    self.image_map[img_name] = os.path.join(folder, img_name)
    
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

transform_pipeline = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Impostato num_workers=1 per evitare blocchi sul sistema
train_loader = DataLoader(NIHChestXrayDataset(train_csv, transform_pipeline), batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
val_loader = DataLoader(NIHChestXrayDataset(val_csv, transform_pipeline), batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

# --- MODELLO ---
print("Caricamento modello biomedico...")
model = AutoModelForImageClassification.from_pretrained(
    "Tsomaros/swin-base-patch4-window7-224_Chest_Xray",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
).to(device)

criterion = nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()

# --- TRAINING LOOP ---
best_auc, patience_counter = 0.0, 0
num_training_steps = (len(train_loader) // ACCUMULATION_STEPS) * EPOCHS

for epoch in range(1, EPOCHS + 1):
    if epoch <= FREEZE_EPOCHS:
        for param in model.swin.parameters(): param.requires_grad = False
        optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
        scheduler = None
    else:
        for param in model.swin.parameters(): param.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
        # Cosine Scheduler con Warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=num_training_steps)

    model.train()
    running_loss = 0.0
    
    for i, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoca {epoch}")):
        images, targets = images.to(device), targets.to(device)
        with torch.cuda.amp.autocast():
            loss = criterion(model(images).logits, targets) / ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler: scheduler.step()
        running_loss += loss.item() * ACCUMULATION_STEPS
    
    # Validazione
    model.eval()
    val_targets, val_outputs = [], []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validazione"):
            outputs = torch.sigmoid(model(images.to(device)).logits)
            val_targets.append(targets.numpy()); val_outputs.append(outputs.cpu().numpy())
    
    y_true, y_pred = np.vstack(val_targets), np.vstack(val_outputs)
    per_class_auc = [roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(num_classes)]
    macro_auc = np.mean(per_class_auc)
    
    print(f"Epoch {epoch} | Loss: {running_loss/len(train_loader):.4f} | Macro-AUC: {macro_auc:.4f}")
    # Report Granulare per la tesi
    for name, auc in zip(classes, per_class_auc):
        print(f"  {name}: {auc:.4f}")
    
    if macro_auc > best_auc:
        best_auc, patience_counter = macro_auc, 0
        torch.save(model.state_dict(), checkpoint_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE: break