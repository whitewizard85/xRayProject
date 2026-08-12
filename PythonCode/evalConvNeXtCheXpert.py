import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
from tqdm import tqdm
import timm

# =====================================================
# 1. CONFIGURAZIONE
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv" 
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_nih_model.pth"

classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
num_classes = len(classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 384
BATCH_SIZE = 16

# =====================================================
# 2. DATASET E TRASFORMAZIONI
# =====================================================
def get_image_path(img_name):
    for i in range(1, 13):
        folder = f"images_{i:03d}"
        path = os.path.join(root_dir, folder, "images", img_name)
        if os.path.exists(path): return path
    return None

class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = get_image_path(img_name)
        if img_path is None or not os.path.exists(img_path):
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), torch.zeros(num_classes)
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        label_vec = torch.zeros(num_classes)
        labels = str(row["Finding Labels"]).split("|")
        for l in labels:
            if l in classes: label_vec[classes.index(l)] = 1.0
        return img, label_vec

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("[DATA] Caricamento dei Dataloader...")
val_loader = DataLoader(NIHChestXrayDataset(val_csv, transform=eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(NIHChestXrayDataset(test_csv, transform=eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# =====================================================
# 3. CARICAMENTO MODELLO
# =====================================================
print("[INIT] Caricamento architettura e pesi...")
model = timm.create_model('convnext_base', pretrained=False, num_classes=num_classes)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# =====================================================
# 4. RUN INFERENCE & REPORTING
# =====================================================
def run_inference(dataloader, label_tag):
    all_targets, all_outputs = [], []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc=f"Inferenza {label_tag}"):
            images = images.to(device)
            outputs = torch.sigmoid(model(images))
            all_targets.append(targets.numpy())
            all_outputs.append(outputs.cpu().numpy())
    return np.vstack(all_targets), np.vstack(all_outputs)

print("\n--- STEP 1: Inferenza ---")
val_targets, val_outputs = run_inference(val_loader, "Validation")
test_targets, test_outputs = run_inference(test_loader, "Test Set")

print("\n--- STEP 2: Ottimizzazione Soglie (Indice di Youden) ---")
best_thresholds = np.zeros(num_classes)
for i in range(num_classes):
    fpr, tpr, thresholds = roc_curve(val_targets[:, i], val_outputs[:, i])
    best_thresholds[i] = thresholds[np.argmax(tpr - fpr)]
    print(f" > {classes[i]:<20} | Soglia: {best_thresholds[i]:.4f}")

# =====================================================
# 5. METRICHE FINALI E ERROR ANALYSIS
# =====================================================
report_data = []
for i in range(num_classes):
    binary_preds = (test_outputs[:, i] >= best_thresholds[i]).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(test_targets[:, i], binary_preds, average='binary', zero_division=0)
    fp = np.sum((binary_preds == 1) & (test_targets[:, i] == 0))
    fn = np.sum((binary_preds == 0) & (test_targets[:, i] == 1))
    auc = roc_auc_score(test_targets[:, i], test_outputs[:, i])
    
    report_data.append({
        "Patologia": classes[i], "ROC-AUC": round(auc, 4), "Precision": round(p, 4), 
        "Recall": round(r, 4), "F1-Score": round(f1, 4), "FP": fp, "FN": fn
    })

df_final = pd.DataFrame(report_data)

# Calcolo e stampa delle medie globali
macro_auc = df_final["ROC-AUC"].mean()
macro_precision = df_final["Precision"].mean()
macro_recall = df_final["Recall"].mean()
macro_f1 = df_final["F1-Score"].mean()

print("\n" + "="*50)
print("📈 SINTESI FINALE (MACRO-AVERAGE)")
print(f"Macro ROC-AUC:    {macro_auc:.4f}")
print(f"Macro Precision:  {macro_precision:.4f}")
print(f"Macro Recall:     {macro_recall:.4f}")
print(f"Macro F1-Score:   {macro_f1:.4f}")
print("="*50 + "\n")

# Salvataggio definitivo
df_final.to_csv("/home/gpuvm/Desktop/Luca Migliaccio/Risultati_ConvNeXt_Test.csv", index=False)
print(df_final.to_string(index=False))
print("\n[FINISH] Report salvato correttamente sul Desktop.")