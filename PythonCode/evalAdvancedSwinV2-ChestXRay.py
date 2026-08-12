import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, precision_recall_curve, auc
from tqdm import tqdm
from transformers import AutoModelForImageClassification

# --- 1. CONFIGURAZIONE ---
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv" 
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
# ASSICURATI che questo percorso punti al tuo file .pth finale
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_swin_biomedical_v2.pth" 

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

val_loader = DataLoader(NIHChestXrayDataset(val_csv, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(NIHChestXrayDataset(test_csv, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- 3. CARICAMENTO MODELLO ---
model = AutoModelForImageClassification.from_pretrained(
    "Tsomaros/swin-base-patch4-window7-224_Chest_Xray",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
).to(device)

# Caricamento pesi
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

def run_inference(dataloader):
    all_targets, all_outputs, all_names = [], [], []
    with torch.no_grad():
        for images, targets, names in tqdm(dataloader, desc="Inference"):
            outputs = torch.sigmoid(model(images.to(device)).logits)
            all_targets.append(targets.numpy())
            all_outputs.append(outputs.cpu().numpy())
            all_names.extend(names)
    return np.vstack(all_targets), np.vstack(all_outputs), all_names

# --- 4. ESECUZIONE ---
print("Avvio inferenza su set di Validazione e Test...")
val_targets, val_outputs, _ = run_inference(val_loader)
test_targets, test_outputs, test_names = run_inference(test_loader)

# --- 5. ANALISI E REPORT ---
print("\n" + "="*95)
print(f"{'Patologia':<20} | {'Soglia':<8} | {'AUC':<8} | {'PR-AUC':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
print("-"*95)

all_preds = np.zeros_like(test_outputs)
auc_scores = []
pr_auc_scores = []
macro_precisions = []
macro_recalls = []
macro_f1s = []

for i in range(num_classes):
    # Calcolo soglia Youden su VAL
    fpr, tpr, thresholds = roc_curve(val_targets[:, i], val_outputs[:, i])
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    
    # Metriche su TEST
    test_auc = roc_auc_score(test_targets[:, i], test_outputs[:, i])
    auc_scores.append(test_auc)
    
    p_c, r_c, _ = precision_recall_curve(test_targets[:, i], test_outputs[:, i])
    test_pr_auc = auc(r_c, p_c)
    pr_auc_scores.append(test_pr_auc)
    
    preds = (test_outputs[:, i] >= best_thresh).astype(int)
    all_preds[:, i] = preds
    
    p, r, f1, _ = precision_recall_fscore_support(test_targets[:, i], preds, average='binary', zero_division=0)
    macro_precisions.append(p)
    macro_recalls.append(r)
    macro_f1s.append(f1)
    
    print(f"{classes[i]:<20} | {best_thresh:.4f}   | {test_auc:.4f}   | {test_pr_auc:.4f}   | {p:.4f}   | {r:.4f}   | {f1:.4f}")

# Medie e Metriche Globali
macro_auc = np.mean(auc_scores)
micro_auc = roc_auc_score(test_targets.ravel(), test_outputs.ravel(), average='micro')
macro_pr_auc = np.mean(pr_auc_scores)
macro_p = np.mean(macro_precisions)
macro_r = np.mean(macro_recalls)
macro_f1 = np.mean(macro_f1s)

print("="*95)
print(f"Media Macro ROC-AUC : {macro_auc:.4f}")
print(f"Media Micro ROC-AUC : {micro_auc:.4f}")
print(f"Macro PR-AUC        : {macro_pr_auc:.4f}")
print(f"Macro Precision     : {macro_p:.4f}")
print(f"Macro Recall        : {macro_r:.4f}")
print(f"Macro F1-Score      : {macro_f1:.4f}")
print("="*95)