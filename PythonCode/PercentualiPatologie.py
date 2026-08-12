import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

# --- CONFIGURAZIONE ---
input_csv = '/home/gpuvm/Desktop/Luca Migliaccio/archive/Data_Entry_2017.csv'
output_file = '/home/gpuvm/Desktop/Luca Migliaccio/distribuzione_patologie_finale_14colori.png'

# --- ELABORAZIONE DATI ---
df = pd.read_csv(input_csv)

# --- CALCOLO PERCENTUALE MULTIPLE PATOLOGIE ---
# Conta quante etichette ci sono per riga separate da '|'
num_labels = df['Finding Labels'].str.split('|').apply(len)
# Consideriamo multiple patologie se il conteggio è > 1 e l'etichetta non è solo "No Finding"
has_multiple = (num_labels > 1) & (df['Finding Labels'] != 'No Finding')

totale_immagini = len(df)
immagini_multiple = has_multiple.sum()
percentuale_multiple = (immagini_multiple / totale_immagini) * 100

print(f"--- ANALISI MULTI-LABEL ---")
print(f"Immagini totali: {totale_immagini}")
print(f"Immagini con multiple patologie: {immagini_multiple}")
print(f"Percentuale calcolata: {percentuale_multiple:.2f}%\n")

# --- PROSECUZIONE GRAFICO ---
all_labels = df['Finding Labels'].str.split('|', expand=True).stack()
label_counts = all_labels.value_counts().drop('No Finding', errors='ignore')

# --- GENERAZIONE COLORI (14 colori distinti da una colormap) ---
colors = cm.tab20(np.linspace(0, 1, len(label_counts)))

# --- GENERAZIONE GRAFICO ---
fig, ax = plt.subplots(figsize=(16, 9))
bars = ax.bar(label_counts.index, label_counts.values, color=colors, edgecolor='black', alpha=0.85)

# Annotazioni
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 100, f'{int(height)}', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Stile
ax.set_title('Distribuzione delle Frequenze delle Patologie (NIH Chest X-ray14)', fontsize=18, fontweight='bold', pad=25)
ax.set_ylabel('Numero di Immagini', fontsize=14, fontweight='bold')
ax.set_xlabel('Patologie', fontsize=14, fontweight='bold')

# Miglioramento etichette X
ax.set_xticks(range(len(label_counts.index)))
ax.set_xticklabels(label_counts.index, rotation=45, ha='right', fontsize=12)

# Griglia e margini
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# Salvataggio
plt.savefig(output_file, dpi=300)
print(f"Grafico professionale con 14 colori salvato in: {output_file}")