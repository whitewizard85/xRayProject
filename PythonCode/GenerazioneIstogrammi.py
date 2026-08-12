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
all_labels = df['Finding Labels'].str.split('|', expand=True).stack()
label_counts = all_labels.value_counts().drop('No Finding', errors='ignore')

# --- GENERAZIONE COLORI (14 colori distinti da una colormap) ---
# Usiamo la colormap 'tab20' che è ideale per dati categorici
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