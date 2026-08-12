import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, confusion_matrix
from tqdm import tqdm
from transformers import AutoModelForImageClassification

# --- 1. CONFIGURAZIONE ---
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv" 
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_swin_biomedical_v2.pth" 

IMAGE_SIZE = 224
BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", 
           "Pleural_Thickening", "Hernia"]
num_classes = len(classes)

# --- 2. DATASET ---
class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.transform = transform
        self.image_map = {f: os.path.join(root_dir, f"images_{i:03d}/images", f) 
                          for i in range(1, 13) for f in os.listdir(os.path.join(root_dir, f"images_{i:03d}", "images"))}
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.image_map.get(row["Image Index"])).convert("RGB")
        if self.transform: img = self.transform(img)
        label_vec = torch.tensor([1.0 if l in str(row["Finding Labels"]).split("|") else 0.0 for l in classes])
        return img, label_vec

transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), 
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

val_loader = DataLoader(NIHChestXrayDataset(val_csv, transform), batch_size=BATCH_SIZE)
test_loader = DataLoader(NIHChestXrayDataset(test_csv, transform), batch_size=BATCH_SIZE)

# --- 3. MODELLO ---
model = AutoModelForImageClassification.from_pretrained("Tsomaros/swin-base-patch4-window7-224_Chest_Xray", 
                                                        num_labels=num_classes, ignore_mismatched_sizes=True).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

def get_predictions(loader):
    targets, outputs = [], []
    with torch.no_grad():
        for imgs, targs in tqdm(loader, desc="Inference"):
            targets.append(targs.numpy())
            outputs.append(torch.sigmoid(model(imgs.to(device)).logits).cpu().numpy())
    return np.vstack(targets), np.vstack(outputs)

val_t, val_o = get_predictions(val_loader)
test_t, test_o = get_predictions(test_loader)

# --- 4. ERROR ANALYSIS ---
print(f"\n{'Patologia':<20} | {'Soglia':<8} | {'FP':<8} | {'FN':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
print("-" * 95)

for i in range(num_classes):
    # Calcolo soglia Youden su Validation
    fpr, tpr, threshs = roc_curve(val_t[:, i], val_o[:, i])
    best_thresh = threshs[np.argmax(tpr - fpr)]
    
    # Classificazione Test
    preds = (test_o[:, i] >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_t[:, i], preds).ravel()
    
    p, r, f1, _ = precision_recall_fscore_support(test_t[:, i], preds, average='binary', zero_division=0)
    print(f"{classes[i]:<20} | {best_thresh:.4f}   | {fp:<8} | {fn:<8} | {p:.4f}   | {r:.4f}   | {f1:.4f}")