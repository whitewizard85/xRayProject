import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.calibration import calibration_curve
import timm
from tqdm import tqdm

# --- 1. CONFIGURAZIONE ---
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_convnext_base_22k.pth"
output_dir = "/home/gpuvm/Desktop/Luca Migliaccio/Analisi_Tesi_Finale"
os.makedirs(output_dir, exist_ok=True)

classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. DATASET ---
def get_image_path(img_name):
    for i in range(1, 13):
        path = os.path.join(root_dir, f"images_{i:03d}", "images", img_name)
        if os.path.exists(path): return path
    return None

class NIHDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.transform = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(get_image_path(row["Image Index"])).convert("RGB")
        label_vec = torch.zeros(len(classes))
        for l in str(row["Finding Labels"]).split("|"):
            if l in classes: label_vec[classes.index(l)] = 1.0
        return self.transform(img), label_vec

# --- 3. INFERENZA ---
model = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device).eval()

loader = DataLoader(NIHDataset(test_csv), batch_size=16, shuffle=False)
all_targets, all_outputs = [], []

print("🚀 Esecuzione inferenza per calibrazione...")
with torch.no_grad():
    for images, targets in tqdm(loader):
        all_targets.append(targets.numpy())
        all_outputs.append(torch.sigmoid(model(images.to(device))).cpu().numpy())

test_targets = np.vstack(all_targets)
test_outputs = np.vstack(all_outputs)

# --- 4. GRAFICO CALIBRAZIONE ---
print("📊 Generazione grafico calibrazione...")
plt.figure(figsize=(12, 10))
ece_list = []

for i in range(len(classes)):
    prob_true, prob_pred = calibration_curve(test_targets[:, i], test_outputs[:, i], n_bins=10)
    ece = np.mean(np.abs(prob_true - prob_pred))
    ece_list.append(ece)
    plt.plot(prob_pred, prob_true, marker='o', label=f"{classes[i]} (ECE: {ece:.3f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Calibrazione Perfetta')
plt.xlabel('Confidenza Media Predetta')
plt.ylabel('Frazione di Positivi (Reale)')
plt.title('Diagrammi di Affidabilità per Classe')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "calibration_plot.png"), dpi=300)
print(f"✅ Fatto. ECE Medio: {np.mean(ece_list):.4f}. Grafico in {output_dir}")