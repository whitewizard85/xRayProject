import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import timm
from tqdm import tqdm

# --- 1. CONFIGURAZIONE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_convnext_base_22k.pth"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
output_dir = "/home/gpuvm/Desktop/Luca Migliaccio/ConvNeXt-Grad-Cam"

# --- 2. MODELLO ---
classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
model = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device).eval()
cam = GradCAM(model=model, target_layers=[model.stages[-1].downsample])

# --- 3. LOGICA DI RICERCA BILANCIATA ---
def get_image_path(img_name):
    for i in range(1, 13):
        path = os.path.join(root_dir, f"images_{i:03d}", "images", img_name)
        if os.path.exists(path): return path
    return None

df = pd.read_csv(test_csv)
transform = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

counts = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
target_class_idx = 4 # Esempio: classe "Mass"

print("🚀 Generazione report bilanciato (10 casi per categoria) con formattazione pulita da tesi...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    if all(v >= 10 for v in counts.values()): break
    
    img_path = get_image_path(row['Image Index'])
    if not img_path: continue
    
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor)).cpu().numpy()[0]
    
    # Valutazione
    gt = 1 if classes[target_class_idx] in str(row['Finding Labels']) else 0
    pred = 1 if output[target_class_idx] >= 0.5 else 0
    
    cat = 'TP' if (gt==1 and pred==1) else 'TN' if (gt==0 and pred==0) else 'FP' if (gt==0 and pred==1) else 'FN'
    if counts[cat] >= 10: continue
    
    # Generazione Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    img_np = np.array(img.resize((384,384))) / 255.0
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    
    # --- PLOT STILIZZATO PER TESI MAGISTRALE (SENZA SCRITTE IN MEZZO) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Pannello Sinistro: Immagine Originale Pulita
    axes[0].imshow(img)
    axes[0].set_title(f"Radiografia di Input\n(Reperto Reale: {row['Finding Labels']})", fontsize=12, fontweight='bold', pad=12)
    axes[0].axis('off') # Rimozione assi in pixel
    
    # Pannello Destro: Grad-CAM Pulita e Formattata (Senza box interni)
    axes[1].imshow(cam_image)
    axes[1].set_title(f"Mappa Grad-CAM | Pred: {classes[target_class_idx]}\nConfidenza: {output[target_class_idx]:.2f} [{cat}]", fontsize=12, fontweight='bold', pad=12)
    axes[1].axis('off') # Rimozione assi in pixel

    cat_dir = os.path.join(output_dir, cat)
    os.makedirs(cat_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(cat_dir, f"{counts[cat]}_{row['Image Index']}.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    counts[cat] += 1

print(f"✅ Fatto! Immagini pulite salvate in {output_dir}")