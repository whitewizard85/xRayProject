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
from transformers import AutoModelForImageClassification
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. CONFIGURAZIONE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
path_swin = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_swin_biomedical_v2.pth"
path_conv = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_convnext_base_22k.pth"

IMAGE_SIZE = 224
classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", 
           "Pleural_Thickening", "Hernia"]
num_classes = len(classes)

# --- 2. DATASET E WRAPPER ---
class NIHDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.transform = transform
        self.image_map = {f: os.path.join(root_dir, f"images_{i:03d}", "images", f) 
                          for i in range(1, 13) for f in os.listdir(os.path.join(root_dir, f"images_{i:03d}", "images")) 
                          if os.path.exists(os.path.join(root_dir, f"images_{i:03d}", "images", f))}
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.image_map.get(row["Image Index"])).convert("RGB")
        if self.transform: img = self.transform(img)
        label_vec = torch.zeros(num_classes)
        for l in str(row["Finding Labels"]).split("|"):
            if l in classes: label_vec[classes.index(l)] = 1.0
        return img, label_vec

class SwinWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        # Restituiamo le feature dell'ultimo blocco encoder per la CAM
        return self.model.swin.encoder.layers[-1].blocks[-1](self.model.swin.embeddings(x))

eval_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_loader = DataLoader(NIHDataset(test_csv, eval_transform), batch_size=1)

# --- 3. MODELLI ---
model_swin = AutoModelForImageClassification.from_pretrained("Tsomaros/swin-base-patch4-window7-224_Chest_Xray", num_labels=num_classes, ignore_mismatched_sizes=True).to(device).eval()
model_swin.load_state_dict(torch.load(path_swin, map_location=device))
model_conv = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=num_classes).to(device).eval()
model_conv.load_state_dict(torch.load(path_conv, map_location=device))

# --- 4. GRAD-CAM (Puntamento corretto) ---
# Usiamo i blocchi che conservano le mappe 2D
target_layers_swin = [model_swin.swin.encoder.layers[-1].blocks[-1]]
target_layers_conv = [model_conv.stages[-1].blocks[-1]]

cam_swin = GradCAMPlusPlus(model=model_swin, target_layers=target_layers_swin)
cam_conv = GradCAMPlusPlus(model=model_conv, target_layers=target_layers_conv)

# --- 5. ESECUZIONE ---
print("Generazione Grad-CAM in corso...")
error_count = 0
for imgs, lbls in tqdm(test_loader):
    imgs, lbls = imgs.to(device), lbls.to(device)
    with torch.no_grad():
        ens_pred = (0.5 * torch.sigmoid(model_swin(imgs).logits) + 0.5 * torch.sigmoid(model_conv(imgs)) > 0.5).float()
    
    if not torch.equal(ens_pred, lbls):
        target_idx = torch.argmax(lbls).item()
        targets = [ClassifierOutputTarget(target_idx)]
        
        # Generazione
        cam_s = cam_swin(input_tensor=imgs, targets=targets)[0]
        cam_c = cam_conv(input_tensor=imgs, targets=targets)[0]
        
        rgb_img = imgs.squeeze().permute(1, 2, 0).cpu().numpy()
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(rgb_img); ax[0].set_title("Originale")
        ax[1].imshow(show_cam_on_image(rgb_img, cam_s)); ax[1].set_title("Swin Grad-CAM")
        ax[2].imshow(show_cam_on_image(rgb_img, cam_c)); ax[2].set_title("ConvNeXt Grad-CAM")
        plt.savefig(f"error_{error_count}.png")
        plt.close()
        error_count += 1
        if error_count >= 10: break

print("Finito. Controlla i file error_X.png nella cartella.")