import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
from tqdm import tqdm
from transformers import SwinForImageClassification, AutoConfig
from rad_dino import RadDino

# --- 1. CONFIGURAZIONE ---
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"

path_rad_dino = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/rad_dino_final.pth"
path_swin = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_swin_biomedical_v2.pth"

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
                          for i in range(1, 13) for f in os.listdir(os.path.join(root_dir, f"images_{i:03d}", "images"))}
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_map.get(row["Image Index"])
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        label_vec = torch.zeros(num_classes)
        for l in str(row["Finding Labels"]).split("|"):
            if l in classes: label_vec[classes.index(l)] = 1.0
        return img, label_vec

eval_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
val_loader = DataLoader(NIHDataset(val_csv, eval_transform), batch_size=BATCH_SIZE)
test_loader = DataLoader(NIHDataset(test_csv, eval_transform), batch_size=BATCH_SIZE)

# --- 3. MODELLI ---
# RAD-DINO
class RAD_DINO_Wrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.rad_dino = RadDino()
        self.head = nn.Linear(768, num_classes)
    def forward(self, x): return self.head(self.rad_dino.model(x).last_hidden_state[:, 0, :])

model_dino = RAD_DINO_Wrapper(num_classes).to(device).eval()
model_dino.load_state_dict(torch.load(path_rad_dino, map_location=device), strict=False)

# SWIN (Configurazione dinamica per evitare mismatch)
config = AutoConfig.from_pretrained("Tsomaros/swin-base-patch4-window7-224_Chest_Xray")
config.num_labels = num_classes
model_swin = SwinForImageClassification(config).to(device).eval()
model_swin.load_state_dict(torch.load(path_swin, map_location=device), strict=False)

# --- 4. INFERENZA ---
def get_probs(loader):
    d_l, s_l, t_l = [], [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Inferenza Ensemble"):
            d_l.append(torch.sigmoid(model_dino(imgs.to(device))).cpu().numpy())
            s_l.append(torch.sigmoid(model_swin(imgs.to(device)).logits).cpu().numpy())
            t_l.append(lbls.numpy())
    return np.vstack(d_l), np.vstack(s_l), np.vstack(t_l)

val_d, val_s, val_t = get_probs(val_loader)
test_d, test_s, test_t = get_probs(test_loader)

# --- 5. ANALISI E REPORT ---
print(f"\n{'Patologia':<20} | {'AUC':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'Alpha'}")
print("-" * 75)

final_probs = np.zeros_like(test_d)
all_preds = np.zeros_like(test_d)
auc_list = []

for j in range(num_classes):
    best_auc, best_alpha = 0, 0.5
    for alpha in np.linspace(0, 1, 21):
        mixed = alpha * val_d[:, j] + (1 - alpha) * val_s[:, j]
        auc = roc_auc_score(val_t[:, j], mixed)
        if auc > best_auc: best_auc, best_alpha = auc, alpha
    
    final_probs[:, j] = best_alpha * test_d[:, j] + (1 - best_alpha) * test_s[:, j]
    
    fpr, tpr, threshs = roc_curve(val_t[:, j], (best_alpha * val_d[:, j] + (1 - best_alpha) * val_s[:, j]))
    best_thresh = threshs[np.argmax(tpr - fpr)]
    all_preds[:, j] = (final_probs[:, j] >= best_thresh).astype(int)
    
    auc_j = roc_auc_score(test_t[:, j], final_probs[:, j])
    auc_list.append(auc_j)
    p, r, f1, _ = precision_recall_fscore_support(test_t[:, j], all_preds[:, j], average='binary', zero_division=0)
    print(f"{classes[j]:<20} | {auc_j:.4f} | {p:.4f} | {r:.4f} | {f1:.4f} | {best_alpha:.2f}")

macro_auc = np.mean(auc_list)
micro_auc = roc_auc_score(test_t.ravel(), final_probs.ravel())
micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(test_t.ravel(), all_preds.ravel(), average='binary')

print("="*75)
print(f"MEDIA MACRO AUC: {macro_auc:.4f} | MEDIA MICRO AUC: {micro_auc:.4f}")
print(f"MEDIA MICRO (P/R/F1): {micro_p:.4f} / {micro_r:.4f} / {micro_f1:.4f}")