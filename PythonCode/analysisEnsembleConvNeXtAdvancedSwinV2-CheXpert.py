import os
import numpy as np
import pandas as pd
import torch
import timm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
from sklearn.calibration import calibration_curve
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
        img_path = self.image_map.get(row["Image Index"])
        img = Image.open(img_path).convert("RGB") if img_path else Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
        if self.transform: img = self.transform(img)
        label_vec = torch.zeros(num_classes)
        for l in str(row["Finding Labels"]).split("|"):
            if l in classes: label_vec[classes.index(l)] = 1.0
        return img, label_vec

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_loader = DataLoader(NIHDataset(val_csv, eval_transform), batch_size=BATCH_SIZE)
test_loader = DataLoader(NIHDataset(test_csv, eval_transform), batch_size=BATCH_SIZE)

# --- 3. MODELLI ---
print("Caricamento modelli in corso...")
model_swin = AutoModelForImageClassification.from_pretrained("Tsomaros/swin-base-patch4-window7-224_Chest_Xray", num_labels=num_classes, ignore_mismatched_sizes=True).to(device).eval()
model_swin.load_state_dict(torch.load(path_swin, map_location=device))
model_conv = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=num_classes).to(device).eval()
model_conv.load_state_dict(torch.load(path_conv, map_location=device))

# --- 4. INFERENZA ---
def get_probs(loader):
    s_l, c_l, t_l = [], [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Inferenza"):
            s_l.append(torch.sigmoid(model_swin(imgs.to(device)).logits).cpu().numpy())
            c_l.append(torch.sigmoid(model_conv(imgs.to(device))).cpu().numpy())
            t_l.append(lbls.numpy())
    return np.vstack(s_l), np.vstack(c_l), np.vstack(t_l)

val_s, val_c, val_t = get_probs(val_loader)
test_s, test_c, test_t = get_probs(test_loader)

# --- 5. ANALISI, REPORT E CALIBRAZIONE ---
print(f"\n{'Patologia':<20} | {'Alpha':<6} | {'AUC':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'Err'}")
print("-" * 85)

all_probs = np.zeros_like(test_s)
all_preds = np.zeros_like(test_s)
metrics_list = []

for j in range(num_classes):
    # Ottimizzazione pesi (Alpha)
    best_auc, best_alpha = 0, 0.5
    for alpha in np.linspace(0, 1, 11):
        mixed = alpha * val_s[:, j] + (1 - alpha) * val_c[:, j]
        auc = roc_auc_score(val_t[:, j], mixed)
        if auc > best_auc: best_auc, best_alpha = auc, alpha
    
    mixed_val = best_alpha * val_s[:, j] + (1 - best_alpha) * val_c[:, j]
    fpr, tpr, threshs = roc_curve(val_t[:, j], mixed_val)
    best_thresh = threshs[np.argmax(tpr - fpr)]
    
    all_probs[:, j] = best_alpha * test_s[:, j] + (1 - best_alpha) * test_c[:, j]
    all_preds[:, j] = (all_probs[:, j] >= best_thresh).astype(int)
    
    auc = roc_auc_score(test_t[:, j], all_probs[:, j])
    p, r, f1, _ = precision_recall_fscore_support(test_t[:, j], all_preds[:, j], average='binary', zero_division=0)
    metrics_list.append([auc, p, r, f1])
    
    print(f"{classes[j]:<20} | {best_alpha:.2f}   | {auc:.4f}   | {p:.4f}   | {r:.4f}   | {f1:.4f}   | {np.sum(test_t[:, j] != all_preds[:, j])}")

# --- 6. MEDIE E GRAFICO CALIBRAZIONE ---
prob_true, prob_pred = calibration_curve(test_t.ravel(), all_probs.ravel(), n_bins=10)
ece = np.mean(np.abs(prob_true - prob_pred))
macro_avg = np.mean(np.array(metrics_list), axis=0)

print("=" * 85)
print(f"MEDIA MACRO AUC: {macro_avg[0]:.4f}")
print(f"ECE (Ensemble): {ece:.4f}")
print("=" * 85)

# Plot
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
plt.plot(prob_pred, prob_true, marker='.', label='Ensemble Swin + ConvNeXt')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title(f'Calibration Curve (ECE: {ece:.4f})')
plt.legend(); plt.grid(True)
plt.savefig("calibration_ensemble.png")
print("Grafico salvato come 'calibration_ensemble.png'.")