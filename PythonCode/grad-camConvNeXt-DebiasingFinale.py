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
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_debiased_convnext.pth"
test_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/test_split.csv"
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
output_dir = "/home/gpuvm/Desktop/Luca Migliaccio/ConvNeXt-Thesis-1000Samples-Complete"

# --- 2. MODELLO E CLASSI ---
classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", 
    "Pleural_Thickening", "Hernia"
]
model = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device).eval()

# Inizializzazione Grad-CAM
cam = GradCAM(model=model, target_layers=[model.stages[-1]])

# --- 3. PIPELINE DI TRASFORMAZIONE ---
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

# Estraiamo 1000 immagini a caso (con seed 42 per coerenza e riproducibilità)
num_samples = min(1000, len(df))
sample_df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)

os.makedirs(output_dir, exist_ok=True)
print(f"🚀 Avvio generazione dei {num_samples} report clinico-visivi completi...")

class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if model_output.ndim == 1:
            return model_output[self.category]
        return model_output[:, self.category]

log_data = []

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    img_name = row['Image Index']
    img_path = get_image_path(img_name)
    if not img_path: 
        continue
     
    try:
        # Pre-elaborazione
        img_pil = Image.open(img_path).convert("RGB")
        input_tensor = transform(img_pil).unsqueeze(0).to(device)
         
        with torch.no_grad():
            output = torch.sigmoid(model(input_tensor)).cpu().numpy()[0]
         
        # Top-1 Prediction
        top1_idx = int(np.argmax(output))
        top1_class_name = classes[top1_idx]
        top1_conf = float(output[top1_idx])

        # Ground Truth reale dal CSV
        gt_str = str(row['Finding Labels'])
        gt_list = [c.strip() for c in gt_str.split('|')]

        # Generazione Grad-CAM basata sulla Top-1
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(top1_idx)])[0, :]
        
        img_np = input_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
         
        # --- CREAZIONE FIGURA A 3 PANNELLI ---
        fig = plt.figure(figsize=(18, 6), dpi=300)
        
        # 1. Immagine originale
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(img_np)
        ax1.axis('off')
        ax1.set_title(f"Original | GT: {gt_str}", fontsize=11, fontweight='bold', color='darkblue')
        
        # 2. Grad-CAM
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(cam_image)
        ax2.axis('off')
        ax2.set_title(f"Grad-CAM (Top-1: {top1_class_name} [{top1_conf:.2f}])", fontsize=11, fontweight='bold', color='darkred')
        
        # 3. Grafico a barre
        ax3 = fig.add_subplot(1, 3, 3)
        colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
        y_pos = np.arange(len(classes))
        
        bars = ax3.barh(y_pos, output, color=colors, align='center', alpha=0.85)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(classes, fontsize=10)
        ax3.invert_yaxis()
        ax3.set_xlabel('Probability (Sigmoid Output)', fontsize=11, fontweight='bold')
        ax3.set_title('Full Class Probabilities Spectrum', fontsize=12, fontweight='bold')
        ax3.set_xlim(0, 1.0)
        ax3.grid(axis='x', linestyle='--', alpha=0.5)

        # Evidenziazione barre
        for c_idx, c_name in enumerate(classes):
            if c_name in gt_list:
                bars[c_idx].set_edgecolor('red')
                bars[c_idx].set_linewidth(2.5)
                bars[c_idx].set_alpha(1.0)
            if c_idx == top1_idx:
                bars[c_idx].set_edgecolor('black')
                bars[c_idx].set_linewidth(2)

        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"sample_{idx+1}_{img_name}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        # Salvataggio log
        log_data.append({
            'Sample_ID': idx+1,
            'Filename': img_name,
            'Ground_Truth': gt_str,
            'Top1_Prediction': top1_class_name,
            'Top1_Confidence': top1_conf
        })
    except Exception as e:
        print(f"⚠️ Errore con l'immagine {img_name}: {e}")
        continue

# Salvataggio del CSV riassuntivo finale
summary_df = pd.DataFrame(log_data)
summary_df.to_csv(os.path.join(output_dir, "summary_1000_samples.csv"), index=False)

print(f"\n✅ Fatto! Elaborazione completata.")
print(f"📁 Immagini e report salvati in: {output_dir}")
print(f"📊 File CSV riassuntivo pronto.")