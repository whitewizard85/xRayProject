import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import timm

# --- 1. CONFIGURAZIONE ---
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv" 
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_swin_v2.pth"
IMAGE_SIZE = 384
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
model = timm.create_model('swinv2_base_window12to24_192to384', pretrained=False, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

def run_inference(dataloader):
    all_targets, all_outputs, all_names = [], [], []
    with torch.no_grad():
        for images, targets, names in tqdm(dataloader, desc="Inference"):
            outputs = torch.sigmoid(model(images.to(device)))
            all_targets.append(targets.numpy()); all_outputs.append(outputs.cpu().numpy()); all_names.extend(names)
    return np.vstack(all_targets), np.vstack(all_outputs), all_names

# --- 4. ANALISI COMPLETA (METRICHE + CALIBRAZIONE + ERRORI) ---
val_targets, val_outputs, _ = run_inference(val_loader)
test_targets, test_outputs, test_names = run_inference(test_loader)

# A. Metriche Performance
print("\n" + "="*85)
print(f"{'Patologia':<20} | {'Soglia':<8} | {'AUC':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
print("-"*85)

all_preds = np.zeros_like(test_outputs)
auc_scores = []
for i in range(num_classes):
    fpr, tpr, thresholds = roc_curve(val_targets[:, i], val_outputs[:, i])
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    test_auc = roc_auc_score(test_targets[:, i], test_outputs[:, i])
    auc_scores.append(test_auc)
    preds = (test_outputs[:, i] >= best_thresh).astype(int)
    all_preds[:, i] = preds
    p, r, f1, _ = precision_recall_fscore_support(test_targets[:, i], preds, average='binary', zero_division=0)
    print(f"{classes[i]:<20} | {best_thresh:.4f}   | {test_auc:.4f}   | {p:.4f}   | {r:.4f}   | {f1:.4f}")

# B. Analisi Calibrazione (ECE)
prob_true, prob_pred = calibration_curve(test_targets.flatten(), test_outputs.flatten(), n_bins=10)
ece = np.mean(np.abs(prob_true - prob_pred))
plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker='o', label=f'Swin_V2 (ECE={ece:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('Confidence Media')
plt.ylabel('Accuratezza Reale')
plt.title('Calibration Curve')
plt.legend(); plt.savefig("calibration_swin.png"); plt.show()

# C. Report Errori Critici (Confidenza > 0.8 ma Errato)
errors = []
for i in range(len(test_names)):
    for c in range(num_classes):
        if test_outputs[i, c] > 0.8 and test_targets[i, c] == 0:
            errors.append({'Image': test_names[i], 'Class': classes[c], 'Confidence': test_outputs[i, c]})
pd.DataFrame(errors).to_csv("error_report_swin.csv", index=False)

print("="*85)
print(f"ANALISI COMPLETATA:")
print(f"1. Media Macro AUC: {np.mean(auc_scores):.4f}")
print(f"2. ECE (Calibrazione): {ece:.4f}")
print(f"3. Casi critici esportati: {len(errors)} in 'error_report_swin.csv'")
print("="*85)