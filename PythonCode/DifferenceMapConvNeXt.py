import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Carica l'immagine originale e quella modificata
orig = Image.open("/home/gpuvm/Desktop/Luca Migliaccio/archive/images_001/images/00000002_000.png").convert('RGB').resize((384,384))
mod_pneumonia = Image.open("/home/gpuvm/Desktop/Luca Migliaccio/Analisi_Tesi_Finale/guided_pneumonia_final.png").convert('RGB')
mod_cardio = Image.open("/home/gpuvm/Desktop/Luca Migliaccio/Analisi_Tesi_Finale/guided_cardiomegaly_final.png").convert('RGB')

# Converti in array
o = np.array(orig).astype(float)
p = np.array(mod_pneumonia).astype(float)
c = np.array(mod_cardio).astype(float)

# Calcola le differenze
diff_p = np.abs(p - o)
diff_c = np.abs(c - o)

# Visualizza
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(diff_p.astype(np.uint8))
ax[0].set_title("Differenza Pneumonia")
ax[1].imshow(diff_c.astype(np.uint8))
ax[1].set_title("Differenza Cardiomegaly")
ax[2].imshow(orig)
ax[2].set_title("Originale")
# Invece di plt.show(), salva il risultato!
output_path = "/home/gpuvm/Desktop/Luca Migliaccio/Analisi_Tesi_Finale/confronto_differenze.png"
plt.savefig(output_path, bbox_inches='tight')
print(f"✅ Confronto salvato in: {output_path}")

import numpy as np

# 'diff_p' e 'diff_c' sono le tue mappe di differenza che hai già
# Calcoliamo le coordinate del centro di massa (dove si concentra l'energia)
def get_center_of_mass(diff_map):
    y, x = np.indices(diff_map.shape[:2])
    weight = diff_map.mean(axis=2) # Somma i canali colore
    center_y = np.sum(y * weight) / np.sum(weight)
    center_x = np.sum(x * weight) / np.sum(weight)
    return center_y, center_x

p_y, p_x = get_center_of_mass(diff_p)
c_y, c_x = get_center_of_mass(diff_c)

print(f"Baricentro Pneumonia: Y={p_y:.2f}, X={p_x:.2f}")
print(f"Baricentro Cardiomegaly: Y={c_y:.2f}, X={c_x:.2f}")