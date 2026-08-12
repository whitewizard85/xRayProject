import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import timm

# --- CONFIGURAZIONE ---
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_debiased_convnext.pth"
output_dir = "/home/gpuvm/Desktop/Luca Migliaccio/Analisi_Calibrazione_Tesi"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "reliability_diagrams"), exist_ok=True)

classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FUNZIONI DI SUPPORTO ---
def get_image_path(img_name):
    for i in range(1, 13):
        path = os.path.join(root_dir, f"images_{i:03d}", "images", img_name)
        if os.path.exists(path): return path
    return None

class NIHDataset(Dataset):
    def __init__(self, csv_file, transform):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(get_image_path(row["Image Index"])).convert("RGB")
        label_vec = torch.zeros(len(classes))
        for l in str(row["Finding Labels"]).split("|"):
            if l in classes: label_vec[classes.index(l)] = 1.0
        return self.transform(img), label_vec

# --- ANALISI AVANZATA DELLA CALIBRAZIONE ---
def perform_calibration_analysis():
    print("🚀 Inizializzazione modello e caricamento pesi...")
    model = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=len(classes))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    transform_clean = transforms.Compose([
        transforms.Resize((384, 384)), transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    loader = DataLoader(NIHDataset(test_csv, transform=transform_clean), batch_size=32, shuffle=False, num_workers=4)
    all_targets, all_outputs = [], []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Esecuzione Inferenza Test Set"):
            all_targets.append(targets.numpy())
            all_outputs.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
    
    y_true = np.vstack(all_targets)
    y_pred_prob = np.vstack(all_outputs)
    y_pred_bin = (y_pred_prob > 0.5).astype(int)

    # 1. Metriche per classe e calibrazione (ECE)
    metrics = []
    print("\n📊 Calcolo metriche di performance e Expected Calibration Error (ECE)...")
    
    for i, cls in enumerate(classes):
        p, r, f, _ = precision_recall_fscore_support(y_true[:, i], y_pred_bin[:, i], average='binary', zero_division=0)
        auc = roc_auc_score(y_true[:, i], y_pred_prob[:, i])
        
        # Calibrazione scikit-learn (reliability curve)
        prob_true, prob_pred = calibration_curve(y_true[:, i], y_pred_prob[:, i], n_bins=10, strategy='uniform')
        
        # Calcolo ECE (Expected Calibration Error) approssimato sui bin
        ece = np.mean(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0.0
        
        metrics.append({
            "Classe": cls, 
            "AUC": round(auc, 4), 
            "Precision": round(p, 4), 
            "Recall": round(r, 4), 
            "F1-Score": round(f, 4), 
            "ECE": round(ece, 4)
        })
        
        # Generazione e salvataggio Reliability Diagram per classe
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfettamente Calibrato")
        plt.plot(prob_pred, prob_true, marker="o", linewidth=2, color="teal", label=f"ConvNeXt ({cls})")
        plt.xlabel("Confidenza Media Predetta", fontsize=11)
        plt.ylabel("Frazione di Positivi (Accuratezza)", fontsize=11)
        plt.title(f"Reliability Diagram - {cls}\n(ECE: {ece:.4f})", fontsize=12, fontweight='bold')
        plt.ylim([-0.05, 1.05])
        plt.xlim([-0.05, 1.05])
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "reliability_diagrams", f"reliability_{cls}.png"), dpi=300)
        plt.close()

    df_metrics = pd.DataFrame(metrics)
    macro_avg = df_metrics[["AUC", "Precision", "Recall", "F1-Score", "ECE"]].mean()
    
    print("\n" + "="*50)
    print(" TABELLA ANALITICA PERFORMANCE & CALIBRAZIONE ")
    print("="*50)
    print(df_metrics.to_string(index=False))
    print("\n" + "-"*50)
    print(" MEDIE MACRO GLOBALI ")
    print("-"*50)
    print(macro_avg.to_string())

    # Salvataggio CSV finale
    csv_path = os.path.join(output_dir, "metriche_calibrazione_dettaglio.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"\n✅ Analisi di calibrazione completata!")
    print(f"📁 Risultati tabulari salvati in: {csv_path}")
    print(f"📈 Reliability diagrams salvati in: {os.path.join(output_dir, 'reliability_diagrams')}")

if __name__ == "__main__":
    perform_calibration_analysis()