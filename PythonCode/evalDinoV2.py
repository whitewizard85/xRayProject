import os
import numpy as np
import pandas as pd
import torch
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
from tqdm import tqdm

# --- 1. CONFIGURAZIONE ---
IMAGE_SIZE = 224
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_model_v2.pth"

classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", 
           "Pleural_Thickening", "Hernia"]
num_classes = len(classes)

# --- 2. DATASET ---
class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.transform = transform
        # Mappatura rapida immagini
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

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_loader = DataLoader(NIHChestXrayDataset(val_csv, transform), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(NIHChestXrayDataset(test_csv, transform), batch_size=BATCH_SIZE, shuffle=False)

# --- 3. MODELLO ---
model = timm.create_model('vit_base_patch14_dinov2.lvd142m', num_classes=num_classes, img_size=IMAGE_SIZE)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device).eval()

def run_inference(dataloader):
    all_targets, all_outputs, all_names = [], [], []
    with torch.no_grad():
        for images, targets, names in tqdm(dataloader, desc="Inference"):
            outputs = torch.sigmoid(model(images.to(device)))
            all_targets.append(targets.numpy())
            all_outputs.append(outputs.cpu().numpy())
            all_names.extend(names)
    return np.vstack(all_targets), np.vstack(all_outputs), all_names

# --- 4. ESECUZIONE ---
print("Esecuzione inferenza...")
val_y, val_pred, _ = run_inference(val_loader)
test_y, test_pred, test_names = run_inference(test_loader)

# --- 5. REPORT ANALITICO COMPLETO ---
print(f"\n{'Patologia':<20} | {'Soglia':<8} | {'AUC':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
print("-" * 80)

thresholds = []
all_preds = np.zeros_like(test_pred)
auc_list = []

for i in range(num_classes):
    # Soglia di Youden
    fpr, tpr, threshs = roc_curve(val_y[:, i], val_pred[:, i])
    best_t = threshs[np.argmax(tpr - fpr)]
    thresholds.append(best_t)
    
    # Metriche test
    test_auc = roc_auc_score(test_y[:, i], test_pred[:, i])
    auc_list.append(test_auc)
    
    preds = (test_pred[:, i] >= best_t).astype(int)
    all_preds[:, i] = preds
    p, r, f1, _ = precision_recall_fscore_support(test_y[:, i], preds, average='binary', zero_division=0)
    
    print(f"{classes[i]:<20} | {best_t:.4f}   | {test_auc:.4f}   | {p:.4f}   | {r:.4f}   | {f1:.4f}")

# Calcolo medie pesate
macro_auc = np.mean(auc_list)
micro_auc = roc_auc_score(test_y.ravel(), test_pred.ravel(), average='micro')
p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(test_y, all_preds, average='macro', zero_division=0)
p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(test_y, all_preds, average='micro', zero_division=0)

print("=" * 80)
print(f"MACRO-AVG | AUC: {macro_auc:.4f} | F1: {f1_macro:.4f}")
print(f"MICRO-AVG | AUC: {micro_auc:.4f} | F1: {f1_micro:.4f}")
print("=" * 80)

# Export per Error Analysis
df_results = pd.DataFrame(test_pred, columns=classes)
df_results['Image Index'] = test_names
df_results.to_csv("test_predictions.csv", index=False)
print("Risultati salvati in 'test_predictions.csv'")