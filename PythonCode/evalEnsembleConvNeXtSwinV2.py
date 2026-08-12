import os
import torch
import pandas as pd
import numpy as np
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    precision_recall_fscore_support, 
    average_precision_score
)

# =====================================================
# 1. CONFIGURAZIONE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
checkpoint_swin = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_swin_v2.pth"
checkpoint_conv = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_convnext_base_22k.pth"

IMAGE_SIZE = 384
BATCH_SIZE = 8
classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", 
           "Pleural_Thickening", "Hernia"]
num_classes = len(classes)

# =====================================================
# 2. DATASET (Con logica di ricerca cartelle)
# =====================================================
class NIHDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = None
        for i in range(1, 13):
            temp_path = os.path.join(self.root_dir, f"images_{i:03d}", "images", img_name)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        if img_path is None: return None, None
        
        image = Image.open(img_path).convert("RGB")
        label = torch.zeros(num_classes)
        for l in str(row["Finding Labels"]).split("|"):
            if l in classes: label[classes.index(l)] = 1.0
        if self.transform: image = self.transform(image)
        return image, label

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0: return torch.empty(0), torch.empty(0)
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_loader = DataLoader(NIHDataset(val_csv, root_dir, transform), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(NIHDataset(test_csv, root_dir, transform), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# =====================================================
# 3. MODELLI & ESTRAZIONE
# =====================================================
def get_model(name, path):
    model = timm.create_model(name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()

model_swin = get_model('swinv2_base_window12to24_192to384', checkpoint_swin)
model_conv = get_model('convnext_base.fb_in22k', checkpoint_conv)

def extract_probs(loader):
    s_list, c_list, t_list = [], [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader):
            if imgs.numel() == 0: continue
            s_list.append(torch.sigmoid(model_swin(imgs.to(device))).cpu().numpy())
            c_list.append(torch.sigmoid(model_conv(imgs.to(device))).cpu().numpy())
            t_list.append(lbls.numpy())
    return np.vstack(s_list), np.vstack(c_list), np.vstack(t_list)

val_s, val_c, val_t = extract_probs(val_loader)
test_s, test_c, test_t = extract_probs(test_loader)

# =====================================================
# 4. OPTIMIZATION & REPORT
# =====================================================
best_alphas, opt_threshs = {}, {}
for j, c in enumerate(classes):
    best_auc = 0
    for alpha in np.arange(0, 1.1, 0.05):
        mixed = alpha * val_s[:, j] + (1 - alpha) * val_c[:, j]
        auc = roc_auc_score(val_t[:, j], mixed)
        if auc > best_auc: best_auc, best_alpha = auc, alpha
    best_alphas[c] = best_alpha
    p, r, t = precision_recall_curve(val_t[:, j], best_alpha * val_s[:, j] + (1 - best_alpha) * val_c[:, j])
    f1 = 2 * (p * r) / (p + r + 1e-8)
    opt_threshs[c] = float(t[np.argmax(f1)])

final_probs = np.zeros_like(test_s)
final_preds = np.zeros_like(test_s)
for j, c in enumerate(classes):
    final_probs[:, j] = best_alphas[c] * test_s[:, j] + (1 - best_alphas[c]) * test_c[:, j]
    final_preds[:, j] = (final_probs[:, j] >= opt_threshs[c]).astype(int)

# Calcolo metriche di dettaglio per classe
class_pr, class_rc, class_f1, _ = precision_recall_fscore_support(
    test_t.astype(int), final_preds.astype(int), average=None, zero_division=0
)

pr_auc_list = []
class_auc_dict = {}
class_pr_dict = {}
class_rc_dict = {}
class_f1_dict = {}

print("\n" + "="*80)
print("TABELLA 1: DETTAGLIO PER CLASSE (SwinV2 + ConvNeXt Ensemble)")
print("="*80)
print(f"{'Patologia':<20} | {'Soglia':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}")
print("-" * 80)

for j, c in enumerate(classes):
    try:
        roc_c = roc_auc_score(test_t[:, j], final_probs[:, j])
    except ValueError:
        roc_c = 0.5000
    
    try:
        pr_c = average_precision_score(test_t[:, j], final_probs[:, j])
    except ValueError:
        pr_c = 0.0000
        
    pr_auc_list.append(pr_c)
    class_auc_dict[c] = roc_c
    class_pr_dict[c] = class_pr[j]
    class_rc_dict[c] = class_rc[j]
    class_f1_dict[c] = class_f1[j]
    
    print(f"{c:<20} | {opt_threshs[c]:<8.4f} | {roc_c:<8.4f} | {pr_c:<8.4f} | {class_pr[j]:<9.4f} | {class_rc[j]:<8.4f} | {class_f1[j]:<8.4f}")

# Calcolo delle 6 metriche globali richieste
macro_roc_auc = np.mean(list(class_auc_dict.values()))
micro_roc_auc = roc_auc_score(test_t.astype(int), final_probs, average='micro')
macro_pr_auc = np.mean(pr_auc_list)
macro_precision = np.mean(list(class_pr_dict.values()))
macro_recall = np.mean(list(class_rc_dict.values()))
macro_f1 = np.mean(list(class_f1_dict.values()))

print("\n" + "="*50)
print("TABELLA 2: LE 6 METRICHE GLOBALI (Ensemble)")
print("="*50)
print(f"{'Metrica Globale':<30} | {'Valore':<10}")
print("-" * 43)
print(f"{'Media Macro ROC-AUC':<30} | {macro_roc_auc:<10.4f}")
print(f"{'Media Micro ROC-AUC':<30} | {micro_roc_auc:<10.4f}")
print(f"{'Macro PR-AUC':<30} | {macro_pr_auc:<10.4f}")
print(f"{'Macro Precision':<30} | {macro_precision:<10.4f}")
print(f"{'Macro Recall':<30} | {macro_recall:<10.4f}")
print(f"{'Macro F1-Score':<30} | {macro_f1:<10.4f}")
print("="*50)