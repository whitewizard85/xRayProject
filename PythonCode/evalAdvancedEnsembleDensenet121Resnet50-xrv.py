import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, precision_recall_fscore_support, auc
import torchxrayvision as xrv

# =====================================================
# CONFIGURAZIONE PATHS
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
# Assumiamo la presenza del validation_split nella stessa directory del test
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv" 
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"

model_path_v4 = "checkpoints/best_densenet121_v4_xrv.pth"
model_path_v5 = "checkpoints/best_resnet50_v5_xrv.pth"

IMAGE_SIZE = 512
classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)

# =====================================================
# DATASET & COLLATE SYSTEM
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
# MODELLI IN MEMORIA
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_densenet = xrv.models.DenseNet(weights="densenet121-res224-all")
class XRVFeatureExtractor(nn.Module):
    def __init__(self, xrv_model, num_classes):
        super().__init__()
        self.features = xrv_model.features            
        self.classifier = nn.Linear(1024, num_classes) 
    def forward(self, x):
        out = F.relu(self.features(x), inplace=True)
        return self.classifier(torch.flatten(F.adaptive_avg_pool2d(out, (1, 1)), 1))

model_v4 = XRVFeatureExtractor(base_densenet, num_classes).to(device)
model_v4.load_state_dict(torch.load(model_path_v4, map_location=device))
model_v4.eval()

base_resnet = xrv.models.ResNet(weights="resnet50-res512-all")
class XRVResNetFeatureExtractor(nn.Module):
    def __init__(self, xrv_resnet, num_classes):
        super().__init__()
        self.base_resnet = xrv_resnet
        self.classifier = nn.Linear(2048, num_classes)
    def forward(self, x): return self.classifier(self.base_resnet.features(x))

model_v5 = XRVResNetFeatureExtractor(base_resnet, num_classes).to(device)
model_v5.load_state_dict(torch.load(model_path_v5, map_location=device))
model_v5.eval()

# =====================================================
# FUNZIONE DI ESTRAZIONE PROBABILITÀ
# =====================================================
def extract_probabilities(csv_path, desc):
    df = pd.read_csv(csv_path)
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    loader = DataLoader(NIHChestDatasetXRV(df, transform), batch_size=16, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    
    probs_v4_all, probs_v5_all, targets_all = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            if images.numel() == 0: continue
            images = images.to(device)
            with torch.cuda.amp.autocast():
                out_v4 = model_v4(images)
                out_v5 = model_v5(images)
            probs_v4_all.append(torch.sigmoid(out_v4).cpu().numpy())
            probs_v5_all.append(torch.sigmoid(out_v5).cpu().numpy())
            targets_all.append(labels.numpy())
            
    return np.vstack(probs_v4_all), np.vstack(probs_v5_all), np.vstack(targets_all)

# --- Estrazione Dati ---
print("1. Estrazione predizioni dal VALIDATION SET per ottimizzazione...")
val_v4, val_v5, val_targets = extract_probabilities(val_csv, "Validation Inference")

print("\n2. Estrazione predizioni dal TEST SET per verifica finale cieca...")
test_v4, test_v5, test_targets = extract_probabilities(test_csv, "Test Inference")

# =====================================================
# GRID SEARCH IPERPARAMETRI SUL VALIDATION SET
# =====================================================
print("\n" + "="*60)
print("AVVIO GRID SEARCH PER CLASSE SUL VALIDATION SET")
print("="*60)

best_alphas = {}
optimized_thresholds = {}
alpha_search_space = np.arange(0.0, 1.05, 0.05) # Da 0.0 a 1.0 con passi di 0.05

for j, c in enumerate(classes):
    best_auc_c = 0.0
    best_alpha_c = 0.5
    best_thresh_c = 0.5
    
    # 1. Trova l'alfa ottimale basato sul ROC-AUC
    for alpha in alpha_search_space:
        mixed_prob_val = alpha * val_v4[:, j] + (1.0 - alpha) * val_v5[:, j]
        try:
            auc_val = roc_auc_score(val_targets[:, j], mixed_prob_val)
            if auc_val > best_auc_c:
                best_auc_c = auc_val
                best_alpha_c = float(alpha)
        except ValueError:
            pass
            
    best_alphas[c] = best_alpha_c
    
    # 2. Con l'alfa migliore, calcola la soglia ottimale massimizzando l'F1-score geometrico
    best_mixed_prob_val = best_alpha_c * val_v4[:, j] + (1.0 - best_alpha_c) * val_v5[:, j]
    try:
        precisions, recalls, thresholds = precision_recall_curve(val_targets[:, j], best_mixed_prob_val)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        if best_idx < len(thresholds):
            best_thresh_c = float(thresholds[best_idx])
    except ValueError:
        best_thresh_c = 0.5
        
    optimized_thresholds[c] = best_thresh_c
    print(f"-> {c:<20} | Best Alpha (v4 weight): {best_alpha_c:.2f} | Opt Thresh: {best_thresh_c:.2f} | Val AUC: {best_auc_c:.4f}")

# =====================================================
# APPLICAZIONE DEI PARAMETRI OTTIMIZZATI SUL TEST SET
# =====================================================
print("\n" + "="*60)
print("EVALUATION FINALE SUL TEST SET CON IPERPARAMETRI OTTIMIZZATI")
print("="*60)

final_test_probs = np.zeros_like(test_v4)
final_test_preds_bin = np.zeros_like(test_v4)
test_auc_list = []
test_pr_auc_list = []
macro_precisions = []
macro_recalls = []
macro_f1s = []
test_auc_dict = {}

print("\n" + "="*95)
print(f"{'Patologia':<20} | {'Soglia':<8} | {'AUC':<8} | {'PR-AUC':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
print("-"*95)

for j, c in enumerate(classes):
    a = best_alphas[c]
    t = optimized_thresholds[c]
    
    # Applichiamo la combinazione lineare customizzata
    final_test_probs[:, j] = a * test_v4[:, j] + (1.0 - a) * test_v5[:, j]
    final_test_preds_bin[:, j] = (final_test_probs[:, j] >= t).astype(int)
    
    try:
        auc_t = roc_auc_score(test_targets[:, j], final_test_probs[:, j])
        test_auc_list.append(auc_t)
        test_auc_dict[c] = float(auc_t)
    except ValueError:
        test_auc_list.append(0.5)
        test_auc_dict[c] = 0.5

    try:
        p_c, r_c, _ = precision_recall_curve(test_targets[:, j], final_test_probs[:, j])
        pr_auc_t = auc(r_c, p_c)
        test_pr_auc_list.append(pr_auc_t)
    except ValueError:
        test_pr_auc_list.append(0.0)

    p, r, f1, _ = precision_recall_fscore_support(test_targets[:, j], final_test_preds_bin[:, j], average='binary', zero_division=0)
    macro_precisions.append(p)
    macro_recalls.append(r)
    macro_f1s.append(f1)

    print(f"{c:<20} | {t:.4f}   | {test_auc_list[-1]:.4f}   | {test_pr_auc_list[-1]:.4f}   | {p:.4f}   | {r:.4f}   | {f1:.4f}")

# Calcolo delle 6 Metriche Globali/Medie
optimized_macro_auc = np.mean(test_auc_list)
micro_auc = roc_auc_score(test_targets.ravel(), final_test_probs.ravel(), average='micro')
macro_pr_auc = np.mean(test_pr_auc_list)
macro_p = np.mean(macro_precisions)
macro_r = np.mean(macro_recalls)
macro_f1 = np.mean(macro_f1s)

print("="*95)
print(f"Media Macro ROC-AUC : {optimized_macro_auc:.4f}")
print(f"Media Micro ROC-AUC : {micro_auc:.4f}")
print(f"Macro PR-AUC        : {macro_pr_auc:.4f}")
print(f"Macro Precision     : {macro_p:.4f}")
print(f"Macro Recall        : {macro_r:.4f}")
print(f"Macro F1-Score      : {macro_f1:.4f}")
print("="*95)

# Esportazione del Checkpoint di Configurazione dell'Ensemble
hyperparameters_output = {
    "optimized_macro_auc_test": float(optimized_macro_auc),
    "best_alphas_per_class": best_alphas,
    "optimized_thresholds_per_class": optimized_thresholds,
    "test_auc_per_class": test_auc_dict
}

with open("checkpoints/hyperoptimized_ensemble_config.json", "w") as f:
    json.dump(hyperparameters_output, f, indent=4)
print("\n[FINISH] Configurazione ottimizzata salvata in: checkpoints/hyperoptimized_ensemble_config.json 🚀")