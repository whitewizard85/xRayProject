import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import torchxrayvision as xrv
import timm  
import optuna

# Disabilitiamo i log troppo invasivi di Optuna per una lettura pulita del terminale
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================
# CONFIGURAZIONE PATHS (PERCORSI ASSOLUTI BLINDATI)
# =====================================================
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv" 
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"

# Percorsi assoluti diretti alla cartella checkpoints in Luca Migliaccio
model_path_v4 = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_densenet121_v4_xrv.pth"
model_path_v5 = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_resnet50_v5_xrv.pth"
model_path_b7 = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_efficientnet_b7_asl.pth"

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)

# =====================================================
# DATASET & GESTIONE IMMAGINI
# =====================================================
def encode_labels(label_str):
    vec = torch.zeros(num_classes)
    labels = str(label_str).split("|")
    for l in labels:
        if l in classes: vec[classes.index(l)] = 1.0
    return vec

def get_image_path(img_name):
    for i in range(1, 13):
        folder = f"images_{i:03d}"
        path = os.path.join(root_dir, folder, "images", img_name)
        if os.path.exists(path): return path
    return None

# 1. Dataset specifico per i modelli TorchXRayVision (v4, v5)
class NIHChestDatasetXRV(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = get_image_path(img_name)
        if img_path is None: return None, None
        try:
            image = Image.open(img_path).convert("L")
            img_np = np.array(image)
            img_np = xrv.datasets.normalize(img_np, maxval=255)
            image = Image.fromarray(img_np)
            label = encode_labels(row["Finding Labels"])
            if self.transform: image = self.transform(image)
            return image, label
        except Exception: return None, None

# 2. Dataset specifico per la EfficientNet-B7 (RGB, ImageNet standard)
class NIHChestDatasetB7(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = get_image_path(img_name)
        if img_path is None: return None, None
        try:
            image = Image.open(img_path).convert("RGB") 
            label = encode_labels(row["Finding Labels"])
            if self.transform: image = self.transform(image)
            return image, label
        except Exception: return None, None

def collate_fn(batch):
    batch = [b for b in batch if b is not None and b[0] is not None]
    if len(batch) == 0: return torch.empty(0), torch.empty(0)
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])

# =====================================================
# MODELLI IN MEMORIA
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo rilevato: {device}")

# Caricamento v4
print("Caricamento modello v4 (DenseNet)...")
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
model_v4.load_state_dict(torch.load(model_path_v4, map_location=device))
model_v4.eval()

# Caricamento v5
print("Caricamento modello v5 (ResNet)...")
base_resnet = xrv.models.ResNet(weights="resnet50-res512-all")
class XRVResNetFeatureExtractor(nn.Module):
    def __init__(self, xrv_resnet, num_classes):
        super().__init__()
        self.base_resnet = xrv_resnet
        self.classifier = nn.Linear(2048, num_classes)
    def forward(self, x): return self.classifier(self.base_resnet.features(x))

model_v5 = XRVResNetFeatureExtractor(base_resnet, num_classes).to(device)
model_v5.load_state_dict(torch.load(model_path_v5, map_location=device))
model_v5.eval()

# Caricamento EfficientNet-B7 con Classificatore Sequenziale Custom
print("Caricamento modello EfficientNet-B7 Custom...")
base_b7 = timm.create_model('efficientnet_b7', pretrained=False, num_classes=num_classes)
# Adattiamo la testa per corrispondere a "classifier.1.weight" dello state_dict
base_b7.classifier = nn.Sequential(
    nn.Identity(),
    nn.Linear(2560, num_classes) # 2560 sono i canali nativi in uscita della B7
)
model_b7 = base_b7.to(device)
model_b7.load_state_dict(torch.load(model_path_b7, map_location=device), strict=False)
model_b7.eval()

# =====================================================
# FUNZIONI DI ESTRAZIONE PROBABILITÀ
# =====================================================
def extract_all_probabilities(csv_path, desc):
    df = pd.read_csv(csv_path)
    
    # 1. Pipeline Estrattiva per v4 e v5 (Grigio, 512x512)
    transform_xrv = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    loader_xrv = DataLoader(NIHChestDatasetXRV(df, transform_xrv), batch_size=16, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    
    probs_v4_all, probs_v5_all, targets_all = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader_xrv, desc=f"{desc} [v4/v5]"):
            if images.numel() == 0: continue
            images = images.to(device)
            with torch.cuda.amp.autocast():
                out_v4 = model_v4(images)
                out_v5 = model_v5(images)
            probs_v4_all.append(torch.sigmoid(out_v4).cpu().numpy())
            probs_v5_all.append(torch.sigmoid(out_v5).cpu().numpy())
            targets_all.append(labels.numpy())
            
    # 2. Pipeline Estrattiva per EfficientNet-B7 (RGB, 600x600)
    transform_b7 = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    loader_b7 = DataLoader(NIHChestDatasetB7(df, transform_b7), batch_size=8, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    
    probs_b7_all = []
    with torch.no_grad():
        for images, _ in tqdm(loader_b7, desc=f"{desc} [B7]"):
            if images.numel() == 0: continue
            images = images.to(device)
            with torch.cuda.amp.autocast():
                out_b7 = model_b7(images)
            probs_b7_all.append(torch.sigmoid(out_b7).cpu().numpy())
            
    return np.vstack(probs_v4_all), np.vstack(probs_v5_all), np.vstack(probs_b7_all), np.vstack(targets_all)

# --- Estrazione Dati ---
print("\n1. Estrazione predizioni dal VALIDATION SET per ottimizzazione a 3 modelli...")
val_v4, val_v5, val_b7, val_targets = extract_all_probabilities(val_csv, "Validation Inference")

print("\n2. Estrazione predizioni dal TEST SET per verifica finale cieca...")
test_v4, test_v5, test_b7, test_targets = extract_all_probabilities(test_csv, "Test Inference")

# Liberiamo la VRAM cancellando i modelli prima di avviare Optuna
del model_v4, model_v5, model_b7
torch.cuda.empty_cache()

# =====================================================
# OTTIMIZZAZIONE BAYESIANA TRIPLA CONGIUNTA (OPTUNA)
# =====================================================
print("\n" + "="*60)
print("AVVIO OTTIMIZZAZIONE BAYESIANA TRIPLA (OPTUNA) PER CLASSE")
print("="*60)

best_w_v4 = {}
best_w_v5 = {}
best_w_b7 = {}
optimized_thresholds = {}

for j, c in enumerate(classes):
    
    def objective(trial):
        w_v4 = trial.suggest_float("w_v4", 0.0, 1.0)
        w_v5 = trial.suggest_float("w_v5", 0.0, 1.0)
        w_b7 = trial.suggest_float("w_b7", 0.0, 1.0)
        threshold = trial.suggest_float("threshold", 0.1, 0.9)
        
        total_w = w_v4 + w_v5 + w_b7
        if total_w == 0: return 0.0
        
        w_v4_n = w_v4 / total_w
        w_v5_n = w_v5 / total_w
        w_b7_n = w_b7 / total_w
        
        mixed_prob_val = w_v4_n * val_v4[:, j] + w_v5_n * val_v5[:, j] + w_b7_n * val_b7[:, j]
        preds_bin = (mixed_prob_val >= threshold).astype(int)
        
        try:
            auc_val = roc_auc_score(val_targets[:, j], mixed_prob_val)
            f1_val = f1_score(val_targets[:, j], preds_bin, zero_division=0)
            return auc_val + 0.5 * f1_val
        except ValueError:
            return 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=300)
    
    best_params = study.best_params
    tot = best_params["w_v4"] + best_params["w_v5"] + best_params["w_b7"]
    
    w4_opt = best_params["w_v4"] / tot
    w5_opt = best_params["w_v5"] / tot
    w7_opt = best_params["w_b7"] / tot
    t_opt = float(best_params["threshold"])
    
    mixed_prob_val_opt = w4_opt * val_v4[:, j] + w5_opt * val_v5[:, j] + w7_opt * val_b7[:, j]
    try:
        auc_val_final = roc_auc_score(val_targets[:, j], mixed_prob_val_opt)
    except ValueError:
        auc_val_final = 0.5
        
    best_w_v4[c] = w4_opt
    best_w_v5[c] = w5_opt
    best_w_b7[c] = w7_opt
    optimized_thresholds[c] = t_opt
    
    print(f"-> {c:<20} | Pesi: v4={w4_opt:.2f}, v5={w5_opt:.2f}, B7={w7_opt:.2f} | Soglia: {t_opt:.2f} | Val AUC: {auc_val_final:.4f}")

# =====================================================
# EVALUATION FINALE SUL TEST SET A 3 MODELLI
# =====================================================
print("\n" + "="*60)
print("EVALUATION FINALE SUL TEST SET CON ENSEMBLE TRIPLO")
print("="*60)

final_test_probs = np.zeros_like(test_v4)
final_test_preds_bin = np.zeros_like(test_v4)
test_auc_list = []
test_auc_dict = {}

for j, c in enumerate(classes):
    w4 = best_w_v4[c]
    w5 = best_w_v5[c]
    w7 = best_w_b7[c]
    t = optimized_thresholds[c]
    
    final_test_probs[:, j] = w4 * test_v4[:, j] + w5 * test_v5[:, j] + w7 * test_b7[:, j]
    final_test_preds_bin[:, j] = (final_test_probs[:, j] >= t).astype(int)
    
    try:
        auc_t = roc_auc_score(test_targets[:, j], final_test_probs[:, j])
        test_auc_list.append(auc_t)
        test_auc_dict[c] = float(auc_t)
    except ValueError:
        test_auc_dict[c] = 0.5

test_targets = test_targets.astype(np.int32)
final_test_preds_bin = final_test_preds_bin.astype(np.int32)

# Estrazione automatica e strutturata di precision, recall e f1 macro dal report
report_dict = classification_report(test_targets, final_test_preds_bin, target_names=classes, zero_division=0, output_dict=True)
print(classification_report(test_targets, final_test_preds_bin, target_names=classes, zero_division=0))

optimized_macro_auc = np.mean(test_auc_list)
optimized_macro_prec = report_dict["macro avg"]["precision"]
optimized_macro_rec = report_dict["macro avg"]["recall"]
optimized_macro_f1 = report_dict["macro avg"]["f1-score"]

print("-"*60)
print("📊 RIEPILOGO METRICHE FINALE SUL TEST SET (MACRO AVERAGES):")
print(f"➔ VECCHIO MACRO AVERAGE ROC-AUC (Grid Search v4+v5):    0.8512")
print(f"➔ NUOVO MACRO AVERAGE ROC-AUC CONGIUNTO (v4+v5+B7):      {optimized_macro_auc:.4f} 🏆")
print(f"➔ NUOVA MACRO PRECISION CONGIUNTA:                      {optimized_macro_prec:.4f}")
print(f"➔ NUOVA MACRO RECALL CONGIUNTA (Capacità Screening):    {optimized_macro_rec:.4f} 🩺")
print(f"➔ NUOVO MACRO F1-SCORE CONGIUNTO:                       {optimized_macro_f1:.4f}")
print("-"*60)

# Esportazione Checkpoint Json Definitivo
hyperparameters_output = {
    "optimized_macro_metrics_test": {
        "macro_roc_auc": float(optimized_macro_auc),
        "macro_precision": float(optimized_macro_prec),
        "macro_recall": float(optimized_macro_rec),
        "macro_f1_score": float(optimized_macro_f1)
    },
    "best_weights_v4": best_w_v4,
    "best_weights_v5": best_w_v5,
    "best_weights_b7": best_w_b7,
    "optimized_thresholds_per_class": optimized_thresholds,
    "test_auc_per_class": test_auc_dict
}

output_json_path = "checkpoints/triplet_ensemble_config.json"
os.makedirs("checkpoints", exist_ok=True)
with open(output_json_path, "w") as f:
    json.dump(hyperparameters_output, f, indent=4)
print(f"\n[FINISH] Configurazione del Triplo Ensemble salvata in: {output_json_path} 🚀")