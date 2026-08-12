import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. CONFIGURAZIONE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 384

# --- 2. IL TUO MODELLO ---
class MultimodalSwin(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.swin = timm.create_model('swinv2_base_window12to24_192to384', pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(self.swin.num_features + 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x, metadata):
        features = self.swin(x)
        return self.classifier(torch.cat((features, metadata), dim=1))

# --- 3. FIX RESHAPE (DINAMICO) ---
def swin_reshape_transform(tensor):
    if len(tensor.shape) == 4: return tensor 
    batch_size, num_patches, channels = tensor.shape
    height = width = int(np.sqrt(num_patches))
    result = tensor.reshape(batch_size, height, width, channels)
    result = result.permute(0, 3, 1, 2)
    return result

class ModelWrapper(nn.Module):
    def __init__(self, model, metadata):
        super().__init__()
        self.model = model
        self.metadata = metadata.to(device)
    def forward(self, x):
        return self.model(x, self.metadata)

# --- 4. CARICAMENTO E ESECUZIONE ---
model = MultimodalSwin(num_classes=14).to(device)
model.load_state_dict(torch.load("/home/gpuvm/Desktop/Luca Migliaccio/checkpoints/best_multimodal_swin.pth", map_location=device))
model.eval()

# --- DEFINIZIONE DI TEST (Quello che mancava) ---
# Sostituisci questi valori con quelli reali presi dal tuo dataset
img_path = "/home/gpuvm/Desktop/Luca Migliaccio/archive/images_001/images/00000001_000.png" # Metti il path reale
img = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
img_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])(img).unsqueeze(0).to(device)

# Creazione meta_tensor di test
meta_tensor = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)

# --- EIGENCAM ---
cam = EigenCAM(
    model=ModelWrapper(model, meta_tensor), 
    target_layers=[model.swin.norm],
    reshape_transform=swin_reshape_transform
)

grayscale_cam = cam(input_tensor=img_tensor)[0]
img_np = np.array(img) / 255.0
visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

plt.imshow(visualization)
plt.savefig("test_output.png")
print("Salvato in test_output.png")