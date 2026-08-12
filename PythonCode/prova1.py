import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torchxrayvision as xrv

# =====================================================
# 1. SETUP & CONFIGURAZIONE
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv" 
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
model_path_v4 = "checkpoints/best_densenet121_v4_xrv.pth"
model_path_v5 = "checkpoints/best_resnet50_v5_xrv.pth"
IMAGE_SIZE = 512
classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
num_classes = len(classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# 2. DEFINIZIONE FUNZIONI E CLASSI
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

# --- Caricamento Modelli ---
class XRVFeatureExtractor(nn.Module):
    def __init__(self, xrv_model, num_classes):
        super().__init__()
        self.features = xrv_model.features            
        self.classifier = nn.Linear(1024, num_classes) 
    def forward(self, x):
        out = F.relu(self.features(x), inplace=True)
        return self.classifier(torch.flatten(F.adaptive_avg_pool2d(out, (1, 1)), 1))

class XRVResNetFeatureExtractor(nn.Module):
    def __init__(self, xrv_resnet, num_classes):
        super().__init__()
        self.base_resnet = xrv_resnet
        self.classifier = nn.Linear(2048, num_classes)
    def forward(self, x): return self.classifier(self.base_resnet.features(x))

base_densenet = xrv.models.DenseNet(weights="densenet121-res224-all")
model_v4 = XRVFeatureExtractor(base_densenet, num_classes).to(device)
model_v4.load_state_dict(torch.load(model_path_v4, map_location=device))
model_v4.eval()

base_resnet = xrv.models.ResNet(weights="resnet50-res512-all")
model_v5 = XRVResNetFeatureExtractor(base_resnet, num_classes).to(device)
model_v5.load_state_dict(torch.load(model_path_v5, map_location=device))
model_v5.eval()

def extract_probabilities(csv_path, desc):
    df = pd.read_csv(csv_path)
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    loader = DataLoader(NIHChestDatasetXRV(df, transform), batch_size=16, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    probs_v4_all, probs_v5_all, targets_all = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            if images.numel() == 0: continue
            images = images.to(device)
            out_v4 = model_v4(images)
            out_v5 = model_v5(images)
            probs_v4_all.append(torch.sigmoid(out_v4).cpu().numpy())
            probs_v5_all.append(torch.sigmoid(out_v5).cpu().numpy())
            targets_all.append(labels.numpy())
    return np.vstack(probs_v4_all), np.vstack(probs_v5_all), np.vstack(targets_all)

# =====================================================
# 3. ESECUZIONE MAIN
# =====================================================
print("--- 1. ESTRAZIONE PROBABILITÀ ---")
val_v4, val_v5, val_targets = extract_probabilities(val_csv, "Validation")
test_v4, test_v5, test_targets = extract_probabilities(test_csv, "Test")

print("\n--- 2. STACKING & EVALUATION ---")
final_test_probs = np.zeros_like(test_v4)
meta_models = {}
auc_scores = []

for j, c in enumerate(classes):
    X_meta = np.column_stack([val_v4[:, j], val_v5[:, j]])
    y_meta = val_targets[:, j]
    meta_clf = LogisticRegression(solver='liblinear', fit_intercept=True)
    meta_clf.fit(X_meta, y_meta)
    meta_models[c] = meta_clf
    
    X_test_meta = np.column_stack([test_v4[:, j], test_v5[:, j]])
    final_test_probs[:, j] = meta_clf.predict_proba(X_test_meta)[:, 1]
    
    auc_scores.append(roc_auc_score(test_targets[:, j], final_test_probs[:, j]))
    print(f"-> {c:<20} | Test AUC: {auc_scores[-1]:.4f}")

# Calcolo soglie ottimali e report
optimized_thresholds = {}
final_preds_bin = np.zeros_like(test_targets)
for j, c in enumerate(classes):
    precisions, recalls, thresholds = precision_recall_curve(val_targets[:, j], 
                                                            meta_models[c].predict_proba(np.column_stack([val_v4[:, j], val_v5[:, j]]))[:, 1])
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
    optimized_thresholds[c] = float(best_thresh)
    final_preds_bin[:, j] = (final_test_probs[:, j] >= best_thresh).astype(int)

print("\n--- REPORT FINALE ---")
print(classification_report(test_targets.astype(int), final_preds_bin, target_names=classes, zero_division=0))
print(f"➔ MACRO AVERAGE ROC-AUC FINALE: {np.mean(auc_scores):.4f}")