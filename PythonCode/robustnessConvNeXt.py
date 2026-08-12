import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
import os

# --- 1. CONFIGURAZIONE ---
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_convnext_base_22k.pth"
output_dir = "/home/gpuvm/Desktop/Luca Migliaccio/Analisi_Tesi_Finale"
os.makedirs(output_dir, exist_ok=True)

classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. DATASET (Necessario per caricare le immagini) ---
def get_image_path(img_name):
    for i in range(1, 13):
        path = os.path.join(root_dir, f"images_{i:03d}", "images", img_name)
        if os.path.exists(path): return path
    return None

class NIHDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = get_image_path(row["Image Index"])
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, torch.zeros(len(classes)) # Non servono i target per l'entropia

# --- 3. TRANSFORMATIONS PER STRESS TEST ---
# --- 3. STRESS TEST TRANSFORMATIONS CORRETTE ---
def get_transform(mode='clean'):
    # 1. Convertiamo in tensore PRIMA di aggiungere rumore o blur
    base = [
        transforms.Resize((384, 384)), 
        transforms.ToTensor(), # Ora x è un tensore, randn_like funzionerà!
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    if mode == 'noise': 
        # Aggiungiamo il rumore dopo ToTensor
        base.insert(2, transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x)))
    if mode == 'blur': 
        # Blur funziona anche su tensori recenti, ma assicuriamoci di averlo dopo ToTensor
        base.insert(2, transforms.GaussianBlur(kernel_size=9))
        
    return transforms.Compose(base)

# --- 4. CARICAMENTO MODELLO ---
model = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device).eval()

# --- 5. ANALISI ROBUSTEZZA ED ENTROPIA ---
print("🚀 Esecuzione Stress Test e calcolo Entropia...")
modes = ['clean', 'noise', 'blur']
results = []

for mode in modes:
    dataset = NIHDataset(test_csv, transform=get_transform(mode))
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    entropies = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc=f"Mode: {mode}"):
            probs = torch.sigmoid(model(images.to(device)))
            # Entropia di Shannon: H = -sum(p * log(p) + (1-p) * log(1-p)) / N_classes
            # Misura quanto il modello è incerto (più è alta, più il modello è confuso)
            h = -(probs * torch.log(probs + 1e-9) + (1-probs) * torch.log(1-probs + 1e-9)).mean(dim=1)
            entropies.extend(h.cpu().numpy())
            
    results.append({"Mode": mode, "Mean_Entropy": np.mean(entropies)})

# --- 6. REPORT ---
df_robust = pd.DataFrame(results)
df_robust.to_csv(os.path.join(output_dir, "robustezza_stress_test.csv"), index=False)
print("\n📊 Risultati Stress Test (Salvati in Analisi_Tesi_Finale):")
print(df_robust)