import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, confusion_matrix
from tqdm import tqdm
from rad_dino import RadDino

# --- 1. CONFIGURAZIONE ---
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/rad_dino_final.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", 
           "Pleural_Thickening", "Hernia"]
num_classes = len(classes)

# --- 2. DATASET E DATALOADER ---
class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.transform = transform
        self.image_map = {img: os.path.join(root_dir, f"images_{i:03d}", "images", img) 
                          for i in range(1, 13) for img in os.listdir(os.path.join(root_dir, f"images_{i:03d}", "images"))}
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

val_loader = DataLoader(NIHChestXrayDataset(val_csv, eval_transform), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(NIHChestXrayDataset(test_csv, eval_transform), batch_size=BATCH_SIZE, shuffle=False)

# --- 3. MODELLO (WRAPPER) ---
class RAD_DINO_Wrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.rad_dino = RadDino()
        self.backbone = self.rad_dino.model
        self.head = nn.Linear(768, num_classes)
        
    def forward(self, x):
        outputs = self.backbone(x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_token)

model = RAD_DINO_Wrapper(num_classes).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# --- 4. INFERENCE ---
def run_inference(dataloader):
    all_targets, all_outputs, all_names = [], [], []
    with torch.no_grad():
        for images, targets, names in tqdm(dataloader, desc="Inference"):
            outputs = torch.sigmoid(model(images.to(device)))
            all_targets.append(targets.numpy())
            all_outputs.append(outputs.cpu().numpy())
            all_names.extend(names)
    return np.vstack(all_targets), np.vstack(all_outputs), all_names

print("Avvio inferenza su set di Validazione e Test...")
val_targets, val_outputs, _ = run_inference(val_loader)
test_targets, test_outputs, test_names = run_inference(test_loader)

# --- 5. ANALISI E REPORT ---
print("\n" + "="*95)
print(f"{'Patologia':<20} | {'Soglia':<8} | {'AUC':<8} | {'TP':<6} | {'FP':<6} | {'TN':<6} | {'FN':<6} | {'F1':<8}")
print("-" * 95)

all_preds = np.zeros_like(test_outputs)
auc_scores = []
error_analysis = []

for i in range(num_classes):
    fpr, tpr, thresholds = roc_curve(val_targets[:, i], val_outputs[:, i])
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    test_auc = roc_auc_score(test_targets[:, i], test_outputs[:, i])
    auc_scores.append(test_auc)
    
    preds = (test_outputs[:, i] >= best_thresh).astype(int)
    all_preds[:, i] = preds
    tn, fp, fn, tp = confusion_matrix(test_targets[:, i], preds).ravel()
    p, r, f1, _ = precision_recall_fscore_support(test_targets[:, i], preds, average='binary', zero_division=0)
    
    errors = np.where((preds != test_targets[:, i]))[0]
    for idx in errors:
        error_analysis.append({"Image": test_names[idx], "Class": classes[i], "Target": test_targets[idx, i], "Pred": preds[idx], "Conf": test_outputs[idx, i]})
    
    print(f"{classes[i]:<20} | {best_thresh:.4f}   | {test_auc:.4f}   | {tp:<6} | {fp:<6} | {tn:<6} | {fn:<6} | {f1:.4f}")

macro_auc = np.mean(auc_scores)
micro_auc = roc_auc_score(test_targets.ravel(), test_outputs.ravel(), average='micro')
p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(test_targets.ravel(), all_preds.ravel(), average='binary')

print("="*95)
print(f"METRICHE AGGREGATE")
print("-" * 95)
print(f"MEDIA MACRO-AUC : {macro_auc:.4f}")
print(f"MEDIA MICRO-AUC : {micro_auc:.4f}")
print(f"MICRO PRECISION : {p_micro:.4f}")
print(f"MICRO RECALL    : {r_micro:.4f}")
print(f"MICRO F1-SCORE  : {f1_micro:.4f}")
print("="*95)

pd.DataFrame(error_analysis).to_csv("error_analysis_rad_dino.csv", index=False)
print("Report errori salvato in 'error_analysis_rad_dino.csv'. Fine valutazione.")