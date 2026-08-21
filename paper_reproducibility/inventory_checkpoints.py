"""
inventory_checkpoints.py

OBIETTIVO
---------
La cartella checkpoints/ contiene la storia completa degli esperimenti
(RAD-DINO, ensemble multimodali, stacking, versioni "extreme"/"final" mai
finite nel paper), mescolata ai modelli che sono davvero nella Tabella 4.
Invece di indovinare quali file mappano a quali righe della tabella, questo
script fa l'inventario COMPLETO e onesto di tutto quello che c'e', con:

  - nome file
  - dimensione (MB)
  - data di ultima modifica (utile per capire l'ordine cronologico degli
    esperimenti e scartare a colpo d'occhio le versioni piu' vecchie)
  - hash SHA-256 (per il manifest di riproducibilita')
  - una colonna vuota "used_in_paper" da compilare A MANO dopo, con il
    nome del modello/riga di tabella corrispondente (o "NO" se non usato)

Copre sia checkpoints/ sia ablation_checkpoints/, e include OGNI file (non
solo .pth), cosi' anche i .json e i .csv di supporto (soglie ottimizzate,
metriche, error analysis) restano visibili nell'inventario.

USO
---
    python inventory_checkpoints.py

OUTPUT
------
results/checkpoint_inventory_FULL.csv -- apri con Excel/LibreOffice/pandas,
ordina per data o per nome, e compila la colonna used_in_paper riga per
riga. Una volta compilata, possiamo filtrare da li' il vero
checkpoint_hashes.csv da mettere nel manifest di riproducibilita'.
"""

import os
import hashlib
import csv
from datetime import datetime

DIRS_TO_SCAN = [
    "/home/gpuvm/Desktop/Luca Migliaccio/checkpoints",
    "/home/gpuvm/Desktop/Luca Migliaccio/ablation_checkpoints",
]
OUTPUT_CSV = "/home/gpuvm/Desktop/Luca Migliaccio/paper_reproducibility/results/checkpoint_inventory_FULL.csv"


def sha256_of_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = []
    for d in DIRS_TO_SCAN:
        if not os.path.isdir(d):
            print(f"[SKIP] {d} non trovata")
            continue
        for fname in sorted(os.listdir(d)):
            path = os.path.join(d, fname)
            if not os.path.isfile(path):
                continue
            size_mb = os.path.getsize(path) / (1024 * 1024)
            mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
            print(f"  hashing {fname} ({size_mb:.1f} MB)...")
            file_hash = sha256_of_file(path)
            rows.append({
                "source_folder": os.path.basename(d),
                "filename": fname,
                "size_MB": round(size_mb, 1),
                "last_modified": mtime,
                "sha256": file_hash,
                "used_in_paper": "",  # da compilare a mano
            })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_folder", "filename", "size_MB", "last_modified", "sha256", "used_in_paper"
        ])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["last_modified"]))

    print(f"\n[DONE] {len(rows)} file inventariati -> {OUTPUT_CSV}")
    print("\n[NEXT STEP] Apri il CSV, ordinato per data di modifica (dal piu' vecchio al piu' recente).")
    print("Per ogni file .pth, compila 'used_in_paper' con il nome del modello/riga di")
    print("tabella corrispondente (es. 'Table 4 - ConvNeXt ImageNet-22k'), o lascia vuoto/")
    print("scrivi 'NO' se non e' un checkpoint usato nel paper. I file .json/.csv di supporto")
    print("possono restare vuoti se non sono checkpoint (li teniamo comunque nell'inventario")
    print("per completezza, ma non serviranno per il manifest finale).")


if __name__ == "__main__":
    main()
