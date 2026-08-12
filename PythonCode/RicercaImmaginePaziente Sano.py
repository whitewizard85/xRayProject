import os

base_path = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
target_filename = "00000002_000.png"
found_path = None

# Ora cerchiamo dentro images_XXX/images/
for i in range(1, 13):
    candidate_path = os.path.join(base_path, f"images_{i:03d}", "images", target_filename)
    if os.path.exists(candidate_path):
        found_path = candidate_path
        print(f"✅ Trovato! Percorso esatto: {found_path}")
        break

if not found_path:
    print("❌ Ancora non trovato. Verifica se il nome file è corretto (es. controlla se è .png o .jpg)")