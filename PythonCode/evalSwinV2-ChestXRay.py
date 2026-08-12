import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, accuracy_score
from tqdm import tqdm
from transformers import AutoModelForImageClassification

# --- 1. CONFIGURAZIONE ---
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv" 
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_swin_biomedical.pth" 
IMAGE_SIZE = 224 
BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", 
           "Pleural_Thickening", "Hernia"]
num_classes = len(classes)

# --- 2. DATASET ---
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
        return img, label_vec, row["Image Index"]

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_loader = DataLoader(NIHChestXrayDataset(val_csv, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(NIHChestXrayDataset(test_csv, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- 3. CARICAMENTO MODELLO ---
model = AutoModelForImageClassification.from_pretrained(
    "Tsomaros/swin-base-patch4-window7-224_Chest_Xray",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
).to(device)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

def run_inference(dataloader):
    all_targets, all_outputs = [], []
    with torch.no_grad():
        for images, targets, _ in tqdm(dataloader, desc="Inference"):
            outputs = torch.sigmoid(model(images.to(device)).logits)
            all_targets.append(targets.numpy())
            all_outputs.append(outputs.cpu().numpy())
    return np.vstack(all_targets), np.vstack(all_outputs)

# --- 4. ANALISI E REPORT ---
val_targets, val_outputs = run_inference(val_loader)
test_targets, test_outputs = run_inference(test_loader)

print(f"\n{'Patologia':<20} | {'Soglia':<8} | {'AUC':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'Errori':<8}")
print("-" * 90)

all_preds = np.zeros_like(test_outputs)
metrics = []

for i in range(num_classes):
    # Calcolo soglia ottimale su VAL
    fpr, tpr, thresholds = roc_curve(val_targets[:, i], val_outputs[:, i])
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    
    # Metriche su TEST
    auc = roc_auc_score(test_targets[:, i], test_outputs[:, i])
    preds = (test_outputs[:, i] >= best_thresh).astype(int)
    all_preds[:, i] = preds
    
    p, r, f1, _ = precision_recall_fscore_support(test_targets[:, i], preds, average='binary', zero_division=0)
    err = np.sum(test_targets[:, i] != preds)
    
    metrics.append([auc, p, r, f1])
    print(f"{classes[i]:<20} | {best_thresh:.4f}   | {auc:.4f}   | {p:.4f}   | {r:.4f}   | {f1:.4f}   | {err}")

# Calcolo Medie
metrics = np.array(metrics)
# Media Macro: media delle colonne
macro_avg = np.mean(metrics, axis=0)

# Media Micro: calcolata su tutti i dati aggregati
micro_auc = roc_auc_score(test_targets, test_outputs, average='micro')
p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(test_targets.ravel(), all_preds.ravel(), average='binary')

print("-" * 90)
print(f"{'MEDIA MACRO':<20} | {'-':<8} | {macro_avg[0]:.4f}   | {macro_avg[1]:.4f}   | {macro_avg[2]:.4f}   | {macro_avg[3]:.4f}   | {'-'}")
print(f"{'MEDIA MICRO':<20} | {'-':<8} | {micro_auc:.4f}   | {p_micro:.4f}   | {r_micro:.4f}   | {f1_micro:.4f}   | {'-'}")