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

# --- 1. CONFIGURAZIONE AGGIORNATA (TESI) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_debiased_convnext.pth"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
# Nuova cartella dedicata e pulita per la tesi
output_dir = "/home/gpuvm/Desktop/Luca Migliaccio/ConvNeXt-Grad-Cam-Thesis-Clean"

# --- 2. MODELLO ---
classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", 
    "Pleural_Thickening", "Hernia"
]
model = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device).eval()

cam = GradCAM(model=model, target_layers=[model.stages[-1]])

# --- 3. PIPELINE DI TRASFORMAZIONE (DEBIASED) ---
transform = transforms.Compose([
    transforms.Resize((420, 420)),
    transforms.CenterCrop((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_image_path(img_name):
    for i in range(1, 13):
        path = os.path.join(root_dir, f"images_{i:03d}", "images", img_name)
        if os.path.exists(path): return path
    return None

df = pd.read_csv(test_csv)
counts = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
target_class_idx = 4  # Classe "Mass"
max_images_per_cat = 50  # 50 immagini per categoria

print(f"🚀 Generazione figure pulite per la tesi (Target: {classes[target_class_idx]}, Max {max_images_per_cat} per categoria)...")
print("-" * 60)

log_data = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    if all(v >= max_images_per_cat for v in counts.values()): 
        break
     
    img_path = get_image_path(row['Image Index'])
    if not img_path: continue
     
    # Pre-elaborazione
    img_pil = Image.open(img_path).convert("RGB")
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
     
    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor)).cpu().numpy()[0]
     
    # Logica di valutazione
    ground_truth_labels = str(row['Finding Labels'])
    gt = 1 if classes[target_class_idx] in ground_truth_labels else 0
    pred = 1 if output[target_class_idx] >= 0.5 else 0
    cat = 'TP' if (gt==1 and pred==1) else 'TN' if (gt==0 and pred==0) else 'FP' if (gt==0 and pred==1) else 'FN'
     
    if counts[cat] >= max_images_per_cat: 
        continue
     
    # Generazione Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    img_np = input_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
     
    # --- PLOT PULITO (ZERO SCRITTE, ALTA RISOLUZIONE) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    
    # Immagine originale
    axes[0].imshow(img_np)
    axes[0].axis('off')
    
    # Mappa Grad-CAM
    axes[1].imshow(cam_image)
    axes[1].axis('off')
    
    # Rimozione dei margini bianchi
    plt.subplots_adjust(wspace=0.02, hspace=0, left=0, right=1, bottom=0, top=1)
    
    cat_dir = os.path.join(output_dir, cat)
    os.makedirs(cat_dir, exist_ok=True)
    
    file_name = f"{counts[cat]}_{row['Image Index']}.png"
    plt.savefig(os.path.join(cat_dir, file_name), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
     
    # Raccolta dati per il report pulito su file
    conf_score = float(output[target_class_idx])
    log_data.append({
        'Category': cat,
        'Filename': file_name,
        'Original_Image': row['Image Index'],
        'Ground_Truth': ground_truth_labels,
        'Confidence': conf_score
    })

    counts[cat] += 1

# Salvataggio del report in un file CSV pulito dentro la cartella di output
summary_df = pd.DataFrame(log_data)
summary_csv_path = os.path.join(output_dir, "summary_report.csv")
summary_df.to_csv(summary_csv_path, index=False)

print("-" * 60)
print(f"✅ Processo completato con successo!")
print(f"📁 Immagini pulite salvate in: {output_dir}")
print(f"📊 Report riepilogativo salvato in: {summary_csv_path}")