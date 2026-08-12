import os
import numpy as np
import pandas as pd
import torch
import timm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. CONFIGURAZIONE ---
BASE_DIR = "/home/gpuvm/Desktop/Luca Migliaccio"
ANALYSIS_DIR = os.path.join(BASE_DIR, "AnalysisSwinV2")
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
# Assicurati che questi percorsi siano corretti per il tuo sistema
test_csv = os.path.join(BASE_DIR, "PythonCode", "test_split.csv")
checkpoint_path = os.path.join(BASE_DIR, "checkpoints", "best_swin_v2.pth")

IMAGE_SIZE = 384
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
        self.image_map = {f: os.path.join(ARCHIVE_DIR, f"images_{i:03d}", "images", f)
                          for i in range(1, 13) for f in os.listdir(os.path.join(ARCHIVE_DIR, f"images_{i:03d}", "images"))
                          if os.path.exists(os.path.join(ARCHIVE_DIR, f"images_{i:03d}", "images", f))}
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_map.get(row["Image Index"])
        img = Image.open(img_path).convert("RGB") if img_path else Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
        if self.transform: img = self.transform(img)
        return img, row["Image Index"]

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. MODELLO ---
model = timm.create_model('swinv2_base_window12to24_192to384', pretrained=False, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Puntamento per EigenCAM (SwinV2 usa model.norm per il layer finale)
target_layers = [model.norm]
cam = EigenCAM(model, target_layers)

# --- 4. ESECUZIONE ---
df_errors = pd.read_csv(os.path.join(ANALYSIS_DIR, "error_report_swin.csv"))
dataset = NIHChestXrayDataset(test_csv, eval_transform)

print("Generazione Eigen-CAM...")
for i in range(min(5, len(df_errors))):
    img_name = df_errors.iloc[i]['Image']
    img_path = dataset.image_map[img_name]
    
    img = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    input_tensor = eval_transform(img).unsqueeze(0).to(device)
    
    # EigenCAM non richiede target_class
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    
    img_float = np.float32(img) / 255.0
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    
    save_path = os.path.join(ANALYSIS_DIR, f"eigencam_error_{i}.png")
    plt.imsave(save_path, cam_image)
    print(f"Salvata: {save_path}")

print("--- Pipeline Eigen-CAM completata ---")