import torch
import timm  # <--- AGGIUNGI QUESTA RIGA!
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Inserisci il path del tuo modello
model_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_convnext_base_22k.pth" 
image_path = "/home/gpuvm/Desktop/Luca Migliaccio/archive/images_001/images/00000002_000.png"

# Caricamento modello (adatta al tuo metodo di load)
model = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=14)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# --- PRE-PROCESSING ---
# Usa le stesse trasformazioni del tuo training
preprocess = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

input_img = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
input_img.requires_grad = True

# --- OTTIMIZZAZIONE (Guided Activation) ---
optimizer = torch.optim.Adam([input_img], lr=0.02)
target_class = 1# Indice per 'Pneumonia'

for i in range(500):
    optimizer.zero_grad()
    output = model(input_img)
    loss = -output[0, target_class]
    loss.backward()
    optimizer.step()

# --- SALVATAGGIO ---
final_img = input_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
# Normalizzazione per visualizzazione
final_img = (final_img - final_img.min()) / (final_img.max() - final_img.min())

save_path = "/home/gpuvm/Desktop/Luca Migliaccio/Analisi_Tesi_Finale/guided_cardiomegaly_final.png"
plt.imsave(save_path, final_img)
print(f"✅ Successo! Immagine salvata in: {save_path}")