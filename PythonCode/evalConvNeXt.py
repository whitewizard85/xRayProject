import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, precision_recall_curve, auc
from tqdm import tqdm
import timm

# =====================================================
# 1. CONFIGURAZIONE PATH E PARAMETRI (Sincronizzati col Train)
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
# NOTA: Assicurati che il nome del file di test sia corretto qui sotto
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv" 
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_convnext_base_22k.pth"

BATCH_SIZE = 16  
IMAGE_SIZE = 384 

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Dispositivo per Inferenza: {device}")

# =====================================================
# 2. DATASET E INFRASTRUTTURA DATI (Nativi del tuo codice)
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
        if self.transform:
            img = self.transform(img)
            
        label_vec = torch.zeros(num_classes)
        labels = str(row["Finding Labels"]).split("|")
        for l in labels:
            if l in classes:
                label_vec[classes.index(l)] = 1.0
                
        return img, label_vec

# Trasformazione pulita (solo Resize e Normalizzazione per Inferenza)
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("[DATA] Caricamento dei Dataloader...")
val_dataset = NIHChestXrayDataset(val_csv, transform=eval_transform)
test_dataset = NIHChestXrayDataset(test_csv, transform=eval_transform)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# =====================================================
# 3. CARICAMENTO MODELLO CONVNEXT (Epoca 2 Blindata)
# =====================================================
print("[INIT] Ripristino Architettura ConvNeXt-Base...")
model = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=num_classes)

print(f"[LOAD] Caricamento dei pesi ottimi registrati da: {checkpoint_path}")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()

# =====================================================
# 4. ESECUZIONE INFERENZA (No Gradienti)
# =====================================================
def run_inference(dataloader, label_tag):
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc=f"Inferenza {label_tag}"):
            images = images.to(device)
            outputs = torch.sigmoid(model(images)) # Applichiamo Sigmoid per predizioni multi-label
            
            all_targets.append(targets.numpy())
            all_outputs.append(outputs.cpu().numpy())
            
    return np.vstack(all_targets), np.vstack(all_outputs)

print("\n--- STEP 1: Estrazione Probabilità sul Validation Set ---")
val_targets, val_outputs = run_inference(val_loader, "Validation")

print("\n--- STEP 2: Estrazione Probabilità sul Test Set Reale ---")
test_targets, test_outputs = run_inference(test_loader, "Test Set")

# =====================================================
# 5. CALCOLO SOGLIE OTTIME (INDICE DI YOUDEN)
# =====================================================
print("\n--- STEP 3: Ottimizzazione Soglie Geometriche (Validation Set) ---")
best_thresholds = np.zeros(num_classes)

for i in range(num_classes):
    fpr, tpr, thresholds = roc_curve(val_targets[:, i], val_outputs[:, i])
    j_scores = tpr - fpr  # Indice di Youden
    best_idx = np.argmax(j_scores)
    best_thresholds[i] = thresholds[best_idx]
    
    # Controllo di sicurezza sulle soglie estreme
    if best_thresholds[i] > 1.0 or best_thresholds[i] < 0.0:
        best_thresholds[i] = 0.5
    print(f" > {classes[i]:<20} | Soglia Ottima Calcolata: {best_thresholds[i]:.4f}")

# =====================================================
# 6. METRICHE FINALI E ERROR ANALYSIS SUL TEST SET
# =====================================================
print("\n=======================================================================")
test_auc_scores = []
test_pr_auc_scores = []
macro_precisions = []
macro_recalls = []
macro_f1s = []
report_data = []

for i in range(num_classes):
    try:
        auc_val = roc_auc_score(test_targets[:, i], test_outputs[:, i])
        test_auc_scores.append(auc_val)
    except ValueError:
        test_auc_scores.append(0.5)

    try:
        p_c, r_c, _ = precision_recall_curve(test_targets[:, i], test_outputs[:, i])
        pr_auc_val = auc(r_c, p_c)
        test_pr_auc_scores.append(pr_auc_val)
    except ValueError:
        test_pr_auc_scores.append(0.0)

    # Applichiamo la soglia di validazione per discretizzare in 0 o 1
    binary_preds = (test_outputs[:, i] >= best_thresholds[i]).astype(int)
    
    # Estrazione Precision, Recall (Sensibilità) e F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_targets[:, i], binary_preds, average='binary', zero_division=0
    )
    
    macro_precisions.append(precision)
    macro_recalls.append(recall)
    macro_f1s.append(f1)
    
    # Calcolo Matrice di Confusione / Error Analysis grezza
    fp = np.sum((binary_preds == 1) & (test_targets[:, i] == 0))
    fn = np.sum((binary_preds == 0) & (test_targets[:, i] == 1))
    
    report_data.append({
        "Patologia": classes[i],
        "Soglia Ottima": round(best_thresholds[i], 4),
        "ROC-AUC Test": round(test_auc_scores[i], 4),
        "PR-AUC Test": round(test_pr_auc_scores[i], 4),
        "Precision": round(precision, 4),
        "Recall (Sens)": round(recall, 4),
        "F1-Score": round(f1, 4),
        "Falsi Positivi (FP)": fp,
        "Falsi Negativi (FN)": fn
    })

# Calcolo delle 6 Metriche Globali/Medie
macro_auc = np.mean(test_auc_scores)
micro_auc = roc_auc_score(test_targets.ravel(), test_outputs.ravel(), average='micro')
macro_pr_auc = np.mean(test_pr_auc_scores)
macro_p = np.mean(macro_precisions)
macro_r = np.mean(macro_recalls)
macro_f1 = np.mean(macro_f1s)

print(f"🎯 VERDETTO FINALE DI TESI: TEST MACRO ROC-AUC = {macro_auc:.4f}")
print("=======================================================================\n")

print("📈 REPORT DIAGNOSTICO ED ERROR ANALYSIS FINALE (TEST SET):")
# Formattazione e stampa pulita a schermo
df_final = pd.DataFrame(report_data)
print("\n", df_final.to_string(index=False))

print("\n" + "="*70)
print(f"Media Macro ROC-AUC : {macro_auc:.4f}")
print(f"Media Micro ROC-AUC : {micro_auc:.4f}")
print(f"Macro PR-AUC        : {macro_pr_auc:.4f}")
print(f"Macro Precision     : {macro_p:.4f}")
print(f"Macro Recall        : {macro_r:.4f}")
print(f"Macro F1-Score      : {macro_f1:.4f}")
print("="*70)

# Salvataggio automatico del report in CSV pronto per essere inserito nella tesi
output_csv_path = "/home/gpuvm/Desktop/Luca Migliaccio/Risultati_ConvNeXt_Test.csv"
df_final.to_csv(output_csv_path, index=False)
print(f"\n[FINISH] Tabella salvata con successo sul tuo desktop in: {output_csv_path} 🚀")