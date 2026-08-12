import timm
import torch

# 1. Definiamo le variabili che mancano
num_classes = 14
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Dispositivo: {device}")
print("[TEST] Tentativo di caricamento modello medicale da Hugging Face...")

try:
    # 2. Carichiamo il modello
    # Assicurati di essere connesso a internet perché lo scaricherà da Hugging Face
    model_name = 'hf_hub:nicolas-metral/convnext-tiny-chexpert'
    
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    
    print(f"✅ Successo! Il modello '{model_name}' è stato caricato correttamente.")
    
    # 3. Verifica veloce: controlla che l'ultimo layer abbia 14 uscite
    print(f"Struttura della testa del modello (deve avere 14 classi):")
    print(model.head)

except Exception as e:
    print(f"❌ Errore durante il caricamento: {e}")