import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, classification_report
import torchxrayvision as xrv

# =====================================================
# CONFIGURAZIONE PATHS E IPERPARAMETRI
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
train_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv" 
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"

model_path_v4 = "checkpoints/best_densenet121_v4_xrv.pth"
model_path_v5 = "checkpoints/best_resnet50_v5_xrv.pth"

IMAGE_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 8  
LR = 1e-3

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)

# =====================================================
# DATASET & PIPELINE
# =====================================================
def encode_labels(label_str):
    vec = torch.zeros(num_classes)
    labels = str(label_str).split("|")
    for l in labels:
        if l in classes: vec[classes.index(l)] = 1.0
    return vec

def get_image_path(img_name):
    for i in range(1, 13):
        folder = f"images_{i:03d}"
        path = os.path.join(root_dir, folder, "images", img_name)
        if os.path.exists(path): return path
    return None

class NIHChestDatasetXRV(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = get_image_path(img_name)
        if img_path is None: return None, None
        try:
            image = Image.open(img_path).convert("L")
            img_np = np.array(image)
            img_np = xrv.datasets.normalize(img_np, maxval=255)
            image = Image.fromarray(img_np)
            label = encode_labels(row["Finding Labels"])
            if self.transform: image = self.transform(image)
            return image, label
        except Exception: return None, None

def collate_fn(batch):
    batch = [b for b in batch if b is not None and b[0] is not None]
    if len(batch) == 0: return torch.empty(0), torch.empty(0)
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])

# =====================================================
# ARCHITETTURA DI FUSIONE AD ALTA STABILITÀ (EARLY FUSION)
# =====================================================
class FeatureFusionEnsemble(nn.Module):
    def __init__(self, path_densenet, path_resnet, num_classes):
        super(FeatureFusionEnsemble, self).__init__()
        
        # 1. Inizializziamo DenseNet e carichiamo i pesi ripuliti
        self.dn_model = xrv.models.DenseNet(weights="densenet121-res224-all")
        dn_state = torch.load(path_densenet, map_location="cpu")
        clean_dn_state = {}
        for k, v in dn_state.items():
            new_k = k.replace("base_model.", "").replace("classifier.", "classifier_old.") 
            clean_dn_state[new_k] = v
        self.dn_model.load_state_dict(clean_dn_state, strict=False)
        
        # Congeliamo i parametri di DenseNet
        for param in self.dn_model.parameters(): param.requires_grad = False
        
        # 2. Inizializziamo ResNet e carichiamo i pesi ripuliti
        self.rn_model = xrv.models.ResNet(weights="resnet50-res512-all")
        rn_state = torch.load(path_resnet, map_location="cpu")
        clean_rn_state = {}
        for k, v in rn_state.items():
            new_k = k.replace("base_model.", "").replace("classifier.", "classifier_old.")
            clean_rn_state[new_k] = v
        self.rn_model.load_state_dict(clean_rn_state, strict=False)
        
        # Congeliamo i parametri di ResNet
        for param in self.rn_model.parameters(): param.requires_grad = False
        
        # 3. La Nuova Testa Neurale Custom (Fonde 1024 + 2048 = 3072 feature)
        self.fusion_classifier = nn.Sequential(
            nn.Linear(1024 + 2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Estrazione feature DenseNet usando il metodo nativo ufficiale della libreria
        # Ritorna [Batch, 1024, H, W] oppure già flattato a seconda della versione, applichiamo pooling di sicurezza
        f_dn = self.dn_model.features(x)
        if len(f_dn.shape) > 2:
            f_dn = F.adaptive_avg_pool2d(f_dn, (1, 1))
            f_dn = torch.flatten(f_dn, 1) # [Batch, 1024]
        
        # Estrazione feature ResNet usando il metodo nativo ufficiale della libreria
        f_rn = self.rn_model.features(x)
        if len(f_rn.shape) > 2:
            f_rn = F.adaptive_avg_pool2d(f_rn, (1, 1))
            f_rn = torch.flatten(f_rn, 1) # [Batch, 2048]
        
        # Concatenazione dei due vettori latenti [Batch, 3072]
        f_combined = torch.cat((f_dn, f_rn), dim=1)
        
        # Classificazione congiunta
        return self.fusion_classifier(f_combined)

# =====================================================
# INIZIALIZZAZIONE ADDESTRAMENTO
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeatureFusionEnsemble(model_path_v4, model_path_v5, num_classes).to(device)

optimizer = torch.optim.AdamW(model.fusion_classifier.parameters(), lr=LR, weight_decay=1e-4)

# Pesi bilanciati per gestire il Class Imbalance
pos_weights = torch.tensor([6.8, 31.6, 6.2, 4.2, 14.2, 12.6, 59.1, 12.9, 18.1, 37.2, 27.6, 42.7, 22.9, 490.0]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

# Loaders
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

train_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

train_loader = DataLoader(NIHChestDatasetXRV(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(NIHChestDatasetXRV(val_df, test_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
test_loader = DataLoader(NIHChestDatasetXRV(test_df, test_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

# =====================================================
# TRAINING LOOP
# =====================================================
print("Inizio addestramento della testa di Feature-Level Fusion... 🚀")
best_val_auc = 0.0

for epoch in range(EPOCHS):
    model.fusion_classifier.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        if images.numel() == 0: continue
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    # Validazione
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            if images.numel() == 0: continue
            images = images.to(device)
            with torch.cuda.amp.autocast():
                outputs = torch.sigmoid(model(images))
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(labels.numpy())
            
    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)
    
    val_aucs = []
    for j in range(num_classes):
        try: val_aucs.append(roc_auc_score(val_targets[:, j], val_preds[:, j]))
        except ValueError: val_aucs.append(0.5)
    macro_val_auc = np.mean(val_aucs)
    
    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Macro-AUC: {macro_val_auc:.4f}")
    
    if macro_val_auc > best_val_auc:
        best_val_auc = macro_val_auc
        torch.save(model.state_dict(), "checkpoints/best_feature_fusion_ensemble.pth")
        print("✔ Nuovo miglior modello salvato!")

# =====================================================
# VALUTAZIONE FINALE SUL TEST SET CIECO
# =====================================================
print("\n" + "="*60)
print("EVALUATION FINALE SUL TEST SET (EARLY FUSION)")
print("="*60)

model.load_state_dict(torch.load("checkpoints/best_feature_fusion_ensemble.pth"))
model.eval()

test_preds, test_targets = [], []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Test Inference"):
        if images.numel() == 0: continue
        images = images.to(device)
        with torch.cuda.amp.autocast():
            outputs = torch.sigmoid(model(images))
        test_preds.append(outputs.cpu().numpy())
        test_targets.append(labels.numpy())

test_preds = np.vstack(test_preds)
test_targets = np.vstack(test_targets)

test_aucs = []
test_preds_bin = (test_preds >= 0.50).astype(int)

for j, c in enumerate(classes):
    try:
        auc = roc_auc_score(test_targets[:, j], test_preds[:, j])
        test_aucs.append(auc)
        print(f"-> {c:<20} | Feature Fusion AUC: {auc:.4f}")
    except ValueError:
        test_aucs.append(0.5)

final_macro_auc = np.mean(test_aucs)
print("\n" + "="*60)
print(f"➔ VECCHIO RECORD MACRO ROC-AUC (Grid Search): 0.8512")
print(f"➔ NUOVO MACRO ROC-AUC (FEATURE-LEVEL FUSION):  {final_macro_auc:.4f}")
print("="*60)
print("\nClassification Report Completo:")
print(classification_report(test_targets.astype(int), test_preds_bin, target_names=classes, zero_division=0))