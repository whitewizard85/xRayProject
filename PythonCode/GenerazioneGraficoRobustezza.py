import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURAZIONE STILE ---
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

scenarios_labels = ['Clean\n(Baseline)', 'Noise\n(Rumore Gaussiano)', 'Blur\n(Sfocatura)']
x = np.arange(len(scenarios_labels))
width = 0.35

# --- PANNELLO 1: CAPACITÀ DISCRIMINATIVA (ROC-AUC e F1-Score) ---
auc_vals = [0.8512, 0.8154, 0.8303]
f1_vals  = [0.2326, 0.2177, 0.2314]

r1 = ax1.bar(x - width/2, auc_vals, width, label='Macro ROC-AUC', color='#2b5c8f', edgecolor='black', linewidth=0.8)
r2 = ax1.bar(x + width/2, f1_vals, width, label='Macro F1-Score', color='#e67e22', edgecolor='black', linewidth=0.8)

ax1.set_ylabel('Valore Metrica', fontsize=11, fontweight='bold')
ax1.set_title('Capacità Discriminativa (Performance)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios_labels, fontsize=10, fontweight='semibold')
ax1.set_ylim(0, 1.0)
ax1.legend(loc='upper right', fontsize=9, frameon=True)

# Annotazioni valori esatti sopra le barre
for bar in r1:
    h = bar.get_height()
    ax1.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in r2:
    h = bar.get_height()
    ax1.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')


# --- PANNELLO 2: INCERTEZZA PREDITTIVA (Entropia di Shannon) ---
entropy_vals = [0.1159, 0.1225, 0.1065]
r3 = ax2.bar(scenarios_labels, entropy_vals, width=0.5, color='#c0392b', edgecolor='black', linewidth=0.8)

ax2.set_ylabel('Entropia Media di Shannon', fontsize=11, fontweight='bold')
ax2.set_title('Incertezza Predittiva del Modello', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(scenarios_labels, fontsize=10, fontweight='semibold')
ax2.set_ylim(0, 0.16)

# Annotazioni valori esatti sopra le barre
for bar in r3:
    h = bar.get_height()
    ax2.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')


# --- SALVATAGGIO ---
plt.tight_layout()
output_dir = "/home/gpuvm/Desktop/Luca Migliaccio/Analisi_Tesi_Robustezza"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "robustezza_metriche_complete.png")
plt.savefig(output_path, bbox_inches='tight')

print(f"Grafico pulito salvato con successo in: {output_path}")
plt.show()