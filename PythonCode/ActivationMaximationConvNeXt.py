import torch
import matplotlib.pyplot as plt
import timm
import os

# --- 1. SETUP (Usa le tue variabili esistenti) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_convnext_base_22k.pth"
# Assicurati che 'classes' sia la stessa lista usata nel training
classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

# --- 2. CARICAMENTO MODELLO ---
model = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval() # Fondamentale: il modello non deve cambiare i pesi

# --- 3. INPUT SYNTHESIS ---
# Creiamo un'immagine casuale (rumore) che il modello inizierà a modificare
# 384x384 è la risoluzione standard del tuo ConvNeXt
input_img = torch.randn(1, 3, 384, 384, requires_grad=True, device=device)

# L'ottimizzatore non agisce sui pesi del modello, ma SOLO sui pixel dell'immagine
optimizer = torch.optim.Adam([input_img], lr=0.1)

target_class = 6 # Esempio: 6 = 'Pneumonia'

print(f"🚀 Sintesi immagine per classe: {classes[target_class]}...")

for i in range(200):
    optimizer.zero_grad()
    
    # Forward pass
    output = model(input_img)
    
    # Loss: vogliamo massimizzare il valore in output per quella classe
    # Usiamo il segno meno perché l'ottimizzatore MINIMIZZA la loss
    loss = -output[0, target_class]
    
    # Backward pass
    loss.backward()
    
    # Aggiornamento: l'ottimizzatore cambia i PIXEL dell'immagine
    optimizer.step()

# --- 4. VISUALIZZAZIONE E SALVATAGGIO ---
final_img = input_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
final_img = (final_img - final_img.min()) / (final_img.max() - final_img.min())

# Salva l'immagine su file invece di usare plt.show()
save_path = os.path.join("/home/gpuvm/Desktop/Luca Migliaccio/Analisi_Tesi_Finale", f"sintesi_{classes[target_class]}.png")
plt.figure(figsize=(6,6))
plt.imshow(final_img)
plt.axis('off')
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
print(f"✅ Immagine salvata in: {save_path}")