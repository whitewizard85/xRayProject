import os
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# =====================================================
# CONFIGURAZIONE PATHS
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
config_json_path = "checkpoints/triplet_ensemble_config.json"
output_analysis_dir = "error_analysis_results"

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)
target_classes_to_analyze = ["Nodule", "Mass", "Pneumonia"]

# =====================================================
# RECUPERO PARAMETRI OTTTIMIZZATI DA JSON
# =====================================================
if not os.path.exists(config_json_path):
    raise FileNotFoundError(f"Impossibile trovare il config JSON in {config_json_path}. Esegui prima l'ensemble!")

with open(config_json_path, "r") as f:
    config = json.load(f)

best_w_v4 = config["best_weights_v4"]
best_w_v5 = config["best_weights_v5"]
best_w_b7 = config["best_weights_b7"]
optimized_thresholds = config["optimized_thresholds_per_class"]

# =====================================================
# RIGENERAZIONE PREDIZIONI SUL TEST SET (SIMULATA VELOCE)
# =====================================================
# Nota: Per evitare di ricaricare i 3 modelli pesanti in VRAM, 
# rieseguiamo l'inferenza solo sulle immagini target per l'analisi visiva.
import torchxrayvision as xrv
import timm
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Funzioni ausiliarie per i percorsi immagini
def get_image_path(img_name):
    for i in range(1, 13):
        folder = f"images_{i:03d}"
        path = os.path.join(root_dir, folder, "images", img_name)
        if os.path.exists(path): return path
    return None

def encode_labels(label_str):
    vec = torch.zeros(num_classes)
    labels = str(label_str).split("|")
    for l in labels:
        if l in classes: vec[classes.index(l)] = 1.0
    return vec

# Definizione Dataset al volo per l'estrazione mirata
class AnalysisDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = get_image_path(img_name)
        return img_name, img_path, encode_labels(row["Finding Labels"])

# Ricarichiamo i modelli solo per l'estrazione finale delle metriche d'errore
print("[INIT] Ricaricamento modelli per estrazione vettori d'errore...")
base_densenet = xrv.models.DenseNet(weights="densenet121-res224-all")
class XRVFeatureExtractor(nn.Module):
    def __init__(self, xrv_model, num_classes):
        super().__init__()
        self.features = xrv_model.features           
        self.classifier = nn.Linear(1024, num_classes) 
    def forward(self, x):
        out = F.relu(self.features(x), inplace=True)
        return self.classifier(torch.flatten(F.adaptive_avg_pool2d(out, (1, 1)), 1))

model_v4 = XRVFeatureExtractor(base_densenet, num_classes).to(device)
model_v4.load_state_dict(torch.load("/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_densenet121_v4_xrv.pth", map_location=device))
model_v4.eval()

base_resnet = xrv.models.ResNet(weights="resnet50-res512-all")
class XRVResNetFeatureExtractor(nn.Module):
    def __init__(self, xrv_resnet, num_classes):
        super().__init__()
        self.base_resnet = xrv_resnet
        self.classifier = nn.Linear(2048, num_classes)
    def forward(self, x): return self.classifier(self.base_resnet.features(x))

model_v5 = XRVResNetFeatureExtractor(base_resnet, num_classes).to(device)
model_v5.load_state_dict(torch.load("/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_resnet50_v5_xrv.pth", map_location=device))
model_v5.eval()

base_b7 = timm.create_model('efficientnet_b7', pretrained=False, num_classes=num_classes)
base_b7.classifier = nn.Sequential(nn.Identity(), nn.Linear(2560, num_classes))
model_b7 = base_b7.to(device)
model_b7.load_state_dict(torch.load("/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_efficientnet_b7_asl.pth", map_location=device), strict=False)
model_b7.eval()

# Trasformazioni dedicate
trans_xrv = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
trans_b7 = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Processamento Test Set
df_test = pd.read_csv(test_csv)
dataset_analysis = AnalysisDataset(df_test)
loader = DataLoader(dataset_analysis, batch_size=1, shuffle=False)

records = []
print("\n[PROCESSING] Calcolo discrepanze e anomalie sui casi del Test Set...")
with torch.no_grad():
    for img_name, img_path, label in tqdm(loader):
        img_path = img_path[0]
        if img_path is None or not os.path.exists(img_path): continue
        
        # Caricamento differenziato per i modelli
        try:
            img_org = Image.open(img_path)
            
            # Input XRV
            img_xrv_np = xrv.datasets.normalize(np.array(img_org.convert("L")), maxval=255)
            t_xrv = trans_xrv(Image.fromarray(img_xrv_np)).unsqueeze(0).to(device)
            
            # Input B7
            t_b7 = trans_b7(img_org.convert("RGB")).unsqueeze(0).to(device)
            
            # Inference
            with torch.cuda.amp.autocast():
                p_v4 = torch.sigmoid(model_v4(t_xrv)).cpu().numpy()[0]
                p_v5 = torch.sigmoid(model_v5(t_xrv)).cpu().numpy()[0]
                p_b7 = torch.sigmoid(model_b7(t_b7)).cpu().numpy()[0]
                
            records.append({
                "img_name": img_name[0],
                "img_path": img_path,
                "label": label.numpy()[0],
                "p_v4": p_v4,
                "p_v5": p_v5,
                "p_b7": p_b7
            })
        except Exception:
            continue

# =====================================================
# COSTRUZIONE ED ESPORTAZIONE REPORT ERRORI VISIVI
# =====================================================
os.makedirs(output_analysis_dir, exist_ok=True)

print("\n[ANALYSIS] Generazione grafici dei Falsi Positivi/Negativi più severi...")

for target_class in target_classes_to_analyze:
    idx_c = classes.index(target_class)
    w4 = best_w_v4[target_class]
    w5 = best_w_v5[target_class]
    w7 = best_w_b7[target_class]
    thresh = optimized_thresholds[target_class]
    
    analysis_list = []
    for r in records:
        # Calcolo probabilità congiunta finale dell'Ensemble Triplo
        prob_ensemble = w4 * r["p_v4"][idx_c] + w5 * r["p_v5"][idx_c] + w7 * r["p_b7"][idx_c]
        true_binary = int(r["label"][idx_c])
        pred_binary = int(prob_ensemble >= thresh)
        
        analysis_list.append({
            "img_name": r["img_name"],
            "img_path": r["img_path"],
            "true_label": true_binary,
            "prob_ensemble": prob_ensemble,
            "pred_label": pred_binary,
            "p_v4": r["p_v4"][idx_c],
            "p_v5": r["p_v5"][idx_c],
            "p_b7": r["p_b7"][idx_c]
        })
        
    df_analysis = pd.DataFrame(analysis_list)
    
    # 1. FALSI POSITIVI CLAMOROSI (Sani ma l'ensemble ha dato una probabilità altissima)
    fps = df_analysis[(df_analysis["true_label"] == 0) & (df_analysis["pred_label"] == 1)]
    fps_worst = fps.sort_values(by="prob_ensemble", ascending=False).head(3)
    
    # 2. FALSI NEGATIVI CLAMOROSI (Malati ma l'ensemble li ha completamente mancati)
    fns = df_analysis[(df_analysis["true_label"] == 1) & (df_analysis["pred_label"] == 0)]
    fns_worst = fns.sort_values(by="prob_ensemble", ascending=True).head(3)
    
    # Funzione per plottare e salvare i dettagli clinici dell'errore
    def plot_errors(df_sub, error_type):
        if df_sub.empty: return
        fig, axes = plt.subplots(1, len(df_sub), figsize=(15, 5))
        if len(df_sub) == 1: axes = [axes]
        
        for i, (_, row) in enumerate(df_sub.iterrows()):
            img = Image.open(row["img_path"]).convert("L")
            axes[i].imshow(img, cmap="gray")
            axes[i].axis("off")
            title_text = (
                f"ID: {row['img_name']}\n"
                f"Prob Ens: {row['prob_ensemble']:.3f} (Soglia: {thresh:.2f})\n"
                f"v4: {row['p_v4']:.2f} | v5: {row['p_v5']:.2f}\n"
                f"B7 (ImageNet): {row['p_b7']:.2f}"
            )
            axes[i].set_title(title_text, fontsize=9, color="darkred" if error_type=="FN" else "darkblue")
            
        plt.suptitle(f"Classe: {target_class} - Primi 3 casi di {error_type} Clamorosi", fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_analysis_dir, f"{target_class}_{error_type}_worst.png"), dpi=200)
        plt.close()

    plot_errors(fps_worst, "FP")
    plot_errors(fns_worst, "FN")

print(f"\n[FINISH] Analisi completata con successo! I grafici strutturati sono pronti nella cartella: {output_analysis_dir}/ 📂")