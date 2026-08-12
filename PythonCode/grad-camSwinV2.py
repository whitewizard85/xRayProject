import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. CONFIGURAZIONE ---
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
val_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/val_split.csv"
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_swin_v2.pth"
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
        self.image_map = {}
        for i in range(1, 13):
            folder = os.path.join(root_dir, f"images_{i:03d}", "images")
            if os.path.exists(folder):
                for img_name in os.listdir(folder):
                    self.image_map[img_name] = os.path.join(folder, img_name)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_map.get(row["Image Index"])
        img = Image.open(img_path).convert("RGB") if img_path else torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        if self.transform: img = self.transform(img)
        label_vec = torch.zeros(num_classes)
        for l in str(row["Finding Labels"]).split("|"):
            if l in classes: label_vec[classes.index(l)] = 1.0
        return img, label_vec, row["Image Index"]

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = NIHChestXrayDataset(test_csv, eval_transform)
test_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
val_loader = DataLoader(NIHChestXrayDataset(val_csv, eval_transform), batch_size=8, shuffle=False, num_workers=0)

# --- 3. MODELLO ---
model = timm.create_model('swinv2_base_window12to24_192to384', pretrained=False, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# --- 4. FUNZIONI DI ANALISI ---
def run_inference(dataloader):
    all_targets, all_outputs, all_names = [], [], []
    with torch.no_grad():
        for images, targets, names in tqdm(dataloader, desc="Inference"):
            outputs = torch.sigmoid(model(images.to(device)))
            all_targets.append(targets.numpy()); all_outputs.append(outputs.cpu().numpy()); all_names.extend(names)
    return np.vstack(all_targets), np.vstack(all_outputs), all_names

def generate_gradcam(image_path, target_class_idx, save_name):
    img = Image.open(image_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    input_tensor = eval_transform(img).unsqueeze(0).to(device)
    cam = GradCAM(model=model, target_layers=[model.norm]) 
    targets = [ClassifierOutputTarget(target_class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    img_float = np.float32(img) / 255.0
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    plt.imsave(f"gradcam_{save_name}.png", cam_image)

# --- 5. ESECUZIONE ---
val_targets, val_outputs, _ = run_inference(val_loader)
test_targets, test_outputs, test_names = run_inference(test_loader)

# Metriche e Report Errori
errors = []
for i in range(len(test_names)):
    for c in range(num_classes):
        if test_outputs[i, c] > 0.8 and test_targets[i, c] == 0:
            errors.append({'Image': test_names[i], 'Class': classes[c], 'Confidence': test_outputs[i, c]})
df_errors = pd.DataFrame(errors)
df_errors.to_csv("error_report_swin.csv", index=False)

# Grad-CAM (3 Errori + 3 Corretti)
print("Generazione Grad-CAM...")
for i in range(min(3, len(df_errors))):
    row = df_errors.iloc[i]
    generate_gradcam(dataset.image_map[row['Image']], classes.index(row['Class']), f"error_{i}")

count = 0
for i in range(len(test_names)):
    for c in range(num_classes):
        if test_outputs[i, c] > 0.8 and test_targets[i, c] == 1 and count < 3:
            generate_gradcam(dataset.image_map[test_names[i]], c, f"correct_{count}")
            count += 1

print("--- Pipeline Completata con successo ---")