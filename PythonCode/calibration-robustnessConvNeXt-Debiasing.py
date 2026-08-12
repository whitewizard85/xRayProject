import os
import torch
import numpy as np
import pandas as pd
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
output_dir = "/home/gpuvm/Desktop/Luca Migliaccio/Analisi_Tesi_Finale"
os.makedirs(output_dir, exist_ok=True)

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

# --- ANALISI COMPLETA ---
def perform_full_analysis():
    model = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=len(classes))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    transform_clean = transforms.Compose([
        transforms.Resize((384, 384)), transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    loader = DataLoader(NIHDataset(test_csv, transform=transform_clean), batch_size=32, shuffle=False)
    all_targets, all_outputs = [], []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Esecuzione Inferenza"):
            all_targets.append(targets.numpy())
            all_outputs.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
    
    y_true = np.vstack(all_targets)
    y_pred_prob = np.vstack(all_outputs)
    y_pred_bin = (y_pred_prob > 0.5).astype(int)

    # 1. Metriche per classe e Medie
    metrics = []
    for i, cls in enumerate(classes):
        p, r, f, _ = precision_recall_fscore_support(y_true[:, i], y_pred_bin[:, i], average='binary', zero_division=0)
        auc = roc_auc_score(y_true[:, i], y_pred_prob[:, i])
        prob_true, prob_pred = calibration_curve(y_true[:, i], y_pred_prob[:, i], n_bins=10)
        ece = np.mean(np.abs(prob_true - prob_pred))
        metrics.append({"Classe": cls, "AUC": auc, "Precision": p, "Recall": r, "F1": f, "ECE": ece})
    
    df_metrics = pd.DataFrame(metrics)
    macro_avg = df_metrics[["AUC", "Precision", "Recall", "F1", "ECE"]].mean()
    
    print("\n--- PERFORMANCE E CALIBRAZIONE ---")
    print(df_metrics.to_string(index=False))
    print(f"\n--- MEDIE GLOBALI ---")
    print(macro_avg.to_string())

    # 2. Robustezza (Entropia)
    modes = {
        'clean': transform_clean,
        'noise': transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(), 
                                   transforms.Lambda(lambda x: x + 0.05*torch.randn_like(x)), 
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'blur': transforms.Compose([transforms.Resize((384, 384)), transforms.GaussianBlur(7), 
                                  transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    
    robust_results = []
    for mode, trans in modes.items():
        loader_s = DataLoader(NIHDataset(test_csv, transform=trans), batch_size=32)
        entropies = []
        with torch.no_grad():
            for images, _ in loader_s:
                probs = torch.clamp(torch.sigmoid(model(images.to(device))), 1e-9, 1-1e-9)
                h = -(probs * torch.log(probs) + (1-probs) * torch.log(1-probs)).mean(dim=1)
                entropies.extend(h.cpu().numpy())
        robust_results.append({"Scenario": mode, "Entropia_Media": np.mean(entropies)})
    
    df_robust = pd.DataFrame(robust_results)
    print("\n--- ANALISI ROBUSTEZZA (ENTROPIA) ---")
    print(df_robust.to_string(index=False))

    # Salvataggio
    df_metrics.to_csv(os.path.join(output_dir, "performance_e_calibrazione.csv"), index=False)
    df_robust.to_csv(os.path.join(output_dir, "robustezza_stress_test.csv"), index=False)
    print(f"\n✅ Analisi completata! File salvati in: {output_dir}")

if __name__ == "__main__":
    perform_full_analysis()