import os
import numpy as np
import pandas as pd
import torch
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_fscore_support, 
    roc_curve, 
    average_precision_score
)
from tqdm import tqdm
from transformers import AutoModelForImageClassification

# --- 1. CONFIGURAZIONE ---
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
path_swin = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_swin_biomedical_v2.pth"
path_conv = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_convnext_base_22k.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", 
           "Pleural_Thickening", "Hernia"]
num_classes = len(classes)

# --- 2. DATASET ---
class NIHDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.transform = transform
        self.image_map = {f: os.path.join(root_dir, f"images_{i:03d}", "images", f) 
                         for i in range(1, 13) for f in os.listdir(os.path.join(root_dir, f"images_{i:03d}", "images")) 
                         if os.path.exists(os.path.join(root_dir, f"images_{i:03d}", "images", f))}
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.image_map.get(row["Image Index"])).convert("RGB")
        if self.transform: img = self.transform(img)
        label_vec = torch.zeros(num_classes)
        for l in str(row["Finding Labels"]).split("|"):
            if l in classes: label_vec[classes.index(l)] = 1.0
        return img, label_vec

eval_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
val_loader = DataLoader(NIHDataset(val_csv, eval_transform), batch_size=BATCH_SIZE)
test_loader = DataLoader(NIHDataset(test_csv, eval_transform), batch_size=BATCH_SIZE)

# --- 3. MODELLI ---
model_swin = AutoModelForImageClassification.from_pretrained("Tsomaros/swin-base-patch4-window7-224_Chest_Xray", num_labels=num_classes, ignore_mismatched_sizes=True).to(device).eval()
model_swin.load_state_dict(torch.load(path_swin, map_location=device))
model_conv = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=num_classes).to(device).eval()
model_conv.load_state_dict(torch.load(path_conv, map_location=device))

# --- 4. INFERENZA ---
def get_probs(loader):
    s_l, c_l, t_l = [], [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader):
            s_l.append(torch.sigmoid(model_swin(imgs.to(device)).logits).cpu().numpy())
            c_l.append(torch.sigmoid(model_conv(imgs.to(device))).cpu().numpy())
            t_l.append(lbls.numpy())
    return np.vstack(s_l), np.vstack(c_l), np.vstack(t_l)

val_s, val_c, val_t = get_probs(val_loader)
test_s, test_c, test_t = get_probs(test_loader)

# --- 5. ANALISI E REPORT ---
print(f"\n" + "="*85)
print("TABELLA 1: DETTAGLIO PER CLASSE")
print("="*85)
print(f"{'Patologia':<20} | {'Soglia':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}")
print("-" * 85)

all_preds = np.zeros_like(test_s)
all_probs = np.zeros_like(test_s)

class_auc_list = []
pr_auc_list = []
class_pr_list = []
class_rc_list = []
class_f1_list = []
best_thresholds = []

for j in range(num_classes):
    best_auc, best_alpha = 0, 0.5
    for alpha in np.linspace(0, 1, 11):
        mixed = alpha * val_s[:, j] + (1 - alpha) * val_c[:, j]
        auc = roc_auc_score(val_t[:, j], mixed)
        if auc > best_auc: best_auc, best_alpha = auc, alpha
    
    mixed_val = best_alpha * val_s[:, j] + (1 - best_alpha) * val_c[:, j]
    fpr, tpr, threshs = roc_curve(val_t[:, j], mixed_val)
    best_thresh = threshs[np.argmax(tpr - fpr)]
    best_thresholds.append(best_thresh)
    
    all_probs[:, j] = best_alpha * test_s[:, j] + (1 - best_alpha) * test_c[:, j]
    all_preds[:, j] = (all_probs[:, j] >= best_thresh).astype(int)
    
    test_auc = roc_auc_score(test_t[:, j], all_probs[:, j])
    pr_auc = average_precision_score(test_t[:, j], all_probs[:, j])
    p, r, f1, _ = precision_recall_fscore_support(test_t[:, j], all_preds[:, j], average='binary', zero_division=0)
    
    class_auc_list.append(test_auc)
    pr_auc_list.append(pr_auc)
    class_pr_list.append(p)
    class_rc_list.append(r)
    class_f1_list.append(f1)
    
    print(f"{classes[j]:<20} | {best_thresh:<8.4f} | {test_auc:<8.4f} | {pr_auc:<8.4f} | {p:<9.4f} | {r:<8.4f} | {f1:<8.4f}")

# --- 6. LE 6 METRICHE GLOBALI ---
macro_roc_auc = np.mean(class_auc_list)
micro_roc_auc = roc_auc_score(test_t.astype(int), all_probs, average='micro')
macro_pr_auc = np.mean(pr_auc_list)
macro_precision = np.mean(class_pr_list)
macro_recall = np.mean(class_rc_list)
macro_f1 = np.mean(class_f1_list)

print("\n" + "="*50)
print("TABELLA 2: LE 6 METRICHE GLOBALI")
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