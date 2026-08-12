import pandas as pd
import os

def check_leakage(train_path, val_path, test_path):
    print("--- Avvio Analisi Data Leakage Completa (Patient-Level) ---")
    
    # Caricamento dei tre split
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    
    # Funzione per estrarre l'ID del paziente dal nome del file (formato NIH)
    def extract_id(row):
        return row['Image Index'].split('_')[0]
    
    train_ids = set(df_train.apply(extract_id, axis=1))
    val_ids = set(df_val.apply(extract_id, axis=1))
    test_ids = set(df_test.apply(extract_id, axis=1))
    
    # Calcolo intersezioni (overlap) tra i vari set
    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)
    
    # Report per la Tesi
    print(f"\n--- NUMERO PAZIENTI ---")
    print(f"Pazienti nel set Training: {len(train_ids)}")
    print(f"Pazienti nel set Validation: {len(val_ids)}")
    print(f"Pazienti nel set Test: {len(test_ids)}")
    
    print(f"\n--- VERIFICA OVERLAP ---")
    print(f"Sovrapposizione Train-Validation: {len(train_val_overlap)}")
    print(f"Sovrapposizione Train-Test: {len(train_test_overlap)}")
    print(f"Sovrapposizione Validation-Test: {len(val_test_overlap)}")
    
    totale_overlap = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
    
    if totale_overlap == 0:
        print("\n✅ RISULTATO: Nessun paziente condiviso tra i set. Split rigoroso confermato.")
    else:
        print(f"\n⚠️ ATTENZIONE: Rilevati pazienti comuni tra i set!")

# Esecuzione
root_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode"
train_csv = os.path.join(root_csv, "train_split.csv")
val_csv = os.path.join(root_csv, "val_split.csv")
test_csv = os.path.join(root_csv, "test_split.csv")

if os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv):
    check_leakage(train_csv, val_csv, test_csv)
else:
    print("❌ Errore: Uno o più file CSV non sono stati trovati nei percorsi specificati.")