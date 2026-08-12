import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm
import timm

# --- CONFIGURAZIONE ---
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_debiased_convnext.pth"
output_dir = "/home/gpuvm/Desktop/Luca Migliaccio/Analisi_Tesi_Robustezza"
os.makedirs(output_dir, exist_ok=True)

classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

# Soglie specifiche per classe estratte dalla tabella di ottimizzazione precedente
class_thresholds = {
    "Atelectasis": 0.0835, "Cardiomegaly": 0.0318, "Effusion": 0.0972, "Infiltration": 0.1439,
    "Mass": 0.0393, "Nodule": 0.0398, "Pneumonia": 0.0134, "Pneumothorax": 0.0295,
    "Consolidation": 0.0392, "Edema": 0.0213, "Emphysema": 0.0181, "Fibrosis": 0.0080,
    "Pleural_Thickening": 0.0207, "Hernia": 0.0028
}

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

# --- ANALISI ROBUSTEZZA E STRESS TEST ---
def perform_robustness_analysis():
    model = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=len(classes))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    # Definizione delle trasformazioni per i tre scenari
    transforms_dict = {
        'clean': transforms.Compose([
            transforms.Resize((384, 384)), transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'noise': transforms.Compose([
            transforms.Resize((384, 384)), transforms.ToTensor(), 
            transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'blur': transforms.Compose([
            transforms.Resize((384, 384)), transforms.GaussianBlur(7), 
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    scenario_results = {}

    for mode, transform in transforms_dict.items():
        print(f"\nEsecuzione inferenza per lo scenario: [{mode.upper()}]")
        loader = DataLoader(NIHDataset(test_csv, transform=transform), batch_size=32, shuffle=False)
        
        all_targets, all_outputs = [], []
        with torch.no_grad():
            for images, targets in tqdm(loader, desc=f"Scenario {mode}"):
                all_targets.append(targets.numpy())
                all_outputs.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
        
        y_true = np.vstack(all_targets)
        y_pred_prob = np.vstack(all_outputs)
        
        # Calcolo Entropia di Shannon per misurare l'incertezza globale dello scenario
        probs_clipped = np.clip(y_pred_prob, 1e-9, 1 - 1e-9)
        entropy_per_sample = -(probs_clipped * np.log(probs_clipped) + (1 - probs_clipped) * np.log(1 - probs_clipped)).mean(axis=1)
        mean_entropy = np.mean(entropy_per_sample)

        # Calcolo AUC per classe usando le soglie specifiche per la binarizzazione
        class_metrics = []
        for i, cls in enumerate(classes):
            auc = roc_auc_score(y_true[:, i], y_pred_prob[:, i])
            
            # Applicazione della soglia specifica associata a ciascuna classe
            th = class_thresholds[cls]
            y_pred_bin = (y_pred_prob[:, i] > th).astype(int)
            p, r, f, _ = precision_recall_fscore_support(y_true[:, i], y_pred_bin, average='binary', zero_division=0)
            
            class_metrics.append({
                "Classe": cls, "Scenario": mode, "ROC-AUC": auc, 
                "Precision": p, "Recall": r, "F1-Score": f
            })
        
        df_class = pd.DataFrame(class_metrics)
        macro_auc = df_class["ROC-AUC"].mean()
        macro_f1 = df_class["F1-Score"].mean()

        scenario_results[mode] = {
            "df_class": df_class,
            "Macro_AUC": macro_auc,
            "Macro_F1": macro_f1,
            "Mean_Entropy": mean_entropy
        }

    # --- AGGREGAZIONE E COMPARAZIONE FINALE ---
    summary_robustness = []
    for mode in transforms_dict.keys():
        summary_robustness.append({
            "Scenario": mode,
            "Macro ROC-AUC": scenario_results[mode]["Macro_AUC"],
            "Macro F1-Score": scenario_results[mode]["Macro_F1"],
            "Entropia Media (Incertezza)": scenario_results[mode]["Mean_Entropy"]
        })

    df_summary = pd.DataFrame(summary_robustness)
    print("\n\n--- TABELLA SINTESI ROBUSTEZZA E STRESS TEST ---")
    print(df_summary.to_string(index=False))

    # Salvataggio dei risultati su CSV per la tesi
    df_summary.to_csv(os.path.join(output_dir, "robustezza_summary_globale.csv"), index=False)
    for mode in transforms_dict.keys():
        scenario_results[mode]["df_class"].to_csv(os.path.join(output_dir, f"robustezza_dettaglio_{mode}.csv"), index=False)

    print(f"\n✅ Analisi di robustezza completata con successo! File salvati in: {output_dir}")

if __name__ == "__main__":
    perform_robustness_analysis()