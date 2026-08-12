import os
import pandas as pd

# Percorso della cartella
chexpert_path = "/home/gpuvm/Desktop/Luca Migliaccio/archiveCheXpert"
# Cerca il file CSV dentro la cartella (cambia il nome se il CSV si chiama diversamente)
csv_files = [f for f in os.listdir(chexpert_path) if f.endswith('.csv')]

if not csv_files:
    print("❌ Non ho trovato nessun file CSV in archiveCheXpert!")
else:
    print(f"✅ Trovati questi CSV: {csv_files}")
    # Leggiamo il primo CSV trovato
    df = pd.read_csv(os.path.join(chexpert_path, csv_files[0]))
    print(f"✅ CSV letto! Il file contiene {len(df)} righe.")
    
    # Controlliamo la prima immagine per vedere se esiste
    # Assumendo che ci sia una colonna chiamata 'Path' o simile
    if 'Path' in df.columns:
        first_img = df['Path'].iloc[0]
        full_path = os.path.join(chexpert_path, first_img.replace("CheXpert-v1.0-small/", ""))
        if os.path.exists(full_path):
            print(f"✅ OK! L'immagine esiste: {full_path}")
        else:
            print(f"❌ Errore: Non trovo l'immagine qui: {full_path}")


    # Leggiamo il CSV di training
df = pd.read_csv(os.path.join(chexpert_path, 'train.csv'))

# Stampiamo tutti i nomi delle colonne
print("--- Nomi delle colonne presenti nel CSV ---")
print(df.columns.tolist())

# Stampiamo le prime 5 righe per vedere come sono formattate le etichette (le patologie)
print("\n--- Esempio di dati (prime 5 righe) ---")
print(df.head())        