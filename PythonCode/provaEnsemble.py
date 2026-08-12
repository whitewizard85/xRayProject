import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import timm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torchxrayvision as xrv

# =====================================================
# 1. CONFIGURAZIONE E WRAPPERS
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
num_classes = 14

class XRV_Fixed_Wrapper(nn.Module):
    def __init__(self, model_type, num_classes):
        super().__init__()
        if model_type == "densenet":
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        else:
            self.model = xrv.models.ResNet(weights="resnet50-res512-all")
        self.features = self.model.features
        self.classifier = nn.Linear(1024 if model_type == "densenet" else 2048, num_classes)
        
    def forward(self, x):
        out = self.features(x)
        if out.dim() == 4:
            out = F.relu(out, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
        return self.classifier(out)

# =====================================================
# 2. DATASET E CARICAMENTO
# =====================================================
class NIHChestDatasetXRV(Dataset):
    def __init__(self, df): self.df = df
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = next((os.path.join(root_dir, f"images_{i:03d}", "images", row["Image Index"]) 
                        for i in range(1, 13) if os.path.exists(os.path.join(root_dir, f"images_{i:03d}", "images", row["Image Index"]))), None)
        if not img_path: return torch.zeros(1, 512, 512), torch.zeros(num_classes)
        img = Image.open(img_path).convert("L")
        img_np = xrv.datasets.normalize(np.array(img), maxval=255)
        return torch.from_numpy(img_np).float().unsqueeze(0), torch.tensor([1.0 if c in str(row["Finding Labels"]).split("|") else 0.0 for c in classes])

test_loader = DataLoader(NIHChestDatasetXRV(pd.read_csv(test_csv)), batch_size=16)

# Caricamento modelli
model_v4 = XRV_Fixed_Wrapper("densenet", 14).to(device).eval()
model_v4.load_state_dict(torch.load("checkpoints/best_densenet121_v4_xrv.pth", map_location=device), strict=False)

model_v5 = XRV_Fixed_Wrapper("resnet", 14).to(device).eval()
model_v5.load_state_dict(torch.load("checkpoints/best_resnet50_v5_xrv.pth", map_location=device), strict=False)

model_cn = timm.create_model('convnext_base', pretrained=False, num_classes=14).to(device).eval()
model_cn.load_state_dict(torch.load("checkpoints/best_convnext_base_22k.pth", map_location=device))

# =====================================================
# 3. INFERENZA E DEBUG
# =====================================================
preds_v4, preds_v5, preds_cn, all_targets = [], [], [], []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Ensemble Debugging"):
        images = images.to(device)
        
        p4 = torch.sigmoid(model_v4(images)).cpu().numpy()
        p5 = torch.sigmoid(model_v5(images)).cpu().numpy()
        pcn = torch.sigmoid(model_cn(F.interpolate(images.repeat(1, 3, 1, 1), size=(384, 384), mode='bilinear'))).cpu().numpy()
        
        preds_v4.append(p4); preds_v5.append(p5); preds_cn.append(pcn)
        all_targets.append(labels.numpy())

t, p4, p5, pcn = np.vstack(all_targets), np.vstack(preds_v4), np.vstack(preds_v5), np.vstack(preds_cn)
ens = (p4 * 0.3) + (p5 * 0.3) + (pcn * 0.4)

def get_macro(pred): return np.mean([roc_auc_score(t[:, j], pred[:, j]) for j in range(num_classes)])

print(f"\n" + "="*30)
print(f"REPORT DIAGNOSTICO MODELLI")
print(f"="*30)
print(f"AUC DenseNet (v4): {get_macro(p4):.4f}")
print(f"AUC ResNet   (v5): {get_macro(p5):.4f}")
print(f"AUC ConvNeXt (cn): {get_macro(pcn):.4f}")
print(f"-"*30)
print(f"AUC ENSEMBLE TOTALE: {get_macro(ens):.4f}")
print(f"="*30)