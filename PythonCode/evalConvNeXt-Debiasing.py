import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, average_precision_score
from tqdm import tqdm
import timm

# --- CONFIGURAZIONE ---
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv" 
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_debiased_convnext.pth"

BATCH_SIZE = 16  
IMAGE_SIZE = 384 

classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
num_classes = len(classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET E TRASFORMAZIONI ---
eval_transform = transforms.Compose([
    transforms.Resize((420, 420)),
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_image_path(img_name):
    for i in range(1, 13):
        path = os.path.join(root_dir, f"images_{i:03d}", "images", img_name)
        if os.path.exists(path): return path
    return None

class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = get_image_path(row["Image Index"])
        if img_path is None: return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), torch.zeros(num_classes)
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        label_vec = torch.zeros(num_classes)
        for l in str(row["Finding Labels"]).split("|"):
            if l in classes: label_vec[classes.index(l)] = 1.0
        return img, label_vec

# --- ESECUZIONE ---
def run_evaluation():
    val_loader = DataLoader(NIHChestXrayDataset(val_csv, transform=eval_transform), batch_size=BATCH_SIZE)
    test_loader = DataLoader(NIHChestXrayDataset(test_csv, transform=eval_transform), batch_size=BATCH_SIZE)

    model = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    def run_inference(loader):
        all_targets, all_outputs = [], []
        with torch.no_grad():
            for imgs, targs in tqdm(loader, desc="Inferenza"):
                all_targets.append(targs.numpy())
                all_outputs.append(torch.sigmoid(model(imgs.to(device))).cpu().numpy())
        return np.vstack(all_targets), np.vstack(all_outputs)

    val_targets, val_outputs = run_inference(val_loader)
    test_targets, test_outputs = run_inference(test_loader)

    # 1. Soglie Ottimali (Youden Index)
    best_thresholds = []
    for i in range(num_classes):
        fpr, tpr, thresh = roc_curve(val_targets[:, i], val_outputs[:, i])
        best_thresholds.append(thresh[np.argmax(tpr - fpr)])

    # 2. Report per Classe
    report_data = []
    preds_bin = np.zeros_like(test_outputs)
    
    class_auc_list = []
    pr_auc_list = []
    class_pr_list = []
    class_rc_list = []
    class_f1_list = []

    for i in range(num_classes):
        preds_bin[:, i] = (test_outputs[:, i] >= best_thresholds[i]).astype(int)
        y_true = test_targets[:, i]
        
        roc_c = roc_auc_score(y_true, test_outputs[:, i])
        pr_c = average_precision_score(y_true, test_outputs[:, i])
        p, r, f, _ = precision_recall_fscore_support(y_true, preds_bin[:, i], average='binary', zero_division=0)
        
        class_auc_list.append(roc_c)
        pr_auc_list.append(pr_c)
        class_pr_list.append(p)
        class_rc_list.append(r)
        class_f1_list.append(f)

        report_data.append({
            "Patologia": classes[i],
            "Soglia": round(best_thresholds[i], 4),
            "ROC-AUC": round(roc_c, 4),
            "PR-AUC": round(pr_c, 4),
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1-Score": round(f, 4),
            "TP": int(np.sum((preds_bin[:, i] == 1) & (y_true == 1))),
            "TN": int(np.sum((preds_bin[:, i] == 0) & (y_true == 0))),
            "FP": int(np.sum((preds_bin[:, i] == 1) & (y_true == 0))),
            "FN": int(np.sum((preds_bin[:, i] == 0) & (y_true == 1)))
        })

    # 3. Calcolo Metriche Globali (Tabella 2)
    macro_roc_auc = np.mean(class_auc_list)
    micro_roc_auc = roc_auc_score(test_targets.astype(int), test_outputs, average='micro')
    macro_pr_auc = np.mean(pr_auc_list)
    macro_precision = np.mean(class_pr_list)
    macro_recall = np.mean(class_rc_list)
    macro_f1 = np.mean(class_f1_list)

    df = pd.DataFrame(report_data)

    # Stampa e Salvataggio
    print("\n" + "="*85)
    print("TABELLA 1: DETTAGLIO PER CLASSE")
    print("="*85)
    print(df[["Patologia", "Soglia", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1-Score"]].to_string(index=False))

    print("\n" + "="*50)
    print("TABELLA 2: LE 6 METRICHE GLOBALI")
    print("="*50)
    print(f"{'Media Macro ROC-AUC':<30} | {macro_roc_auc:.4f}")
    print(f"{'Media Micro ROC-AUC':<30} | {micro_roc_auc:.4f}")
    print(f"{'Macro PR-AUC':<30} | {macro_pr_auc:.4f}")
    print(f"{'Macro Precision':<30} | {macro_precision:.4f}")
    print(f"{'Macro Recall':<30} | {macro_recall:.4f}")
    print(f"{'Macro F1-Score':<30} | {macro_f1:.4f}")
    print("="*50)
    
    df.to_csv("/home/gpuvm/Desktop/Luca Migliaccio/Risultati_Debiased_Test_Completo.csv", index=False)

if __name__ == "__main__":
    run_evaluation()