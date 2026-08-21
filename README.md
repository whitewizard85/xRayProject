# Chest X-Ray Classification — Tesi Migliaccio

## Descrizione

Progetto di classificazione multi-label di radiografie del torace (chest X-ray) tramite
deep learning, sviluppato nell'ambito della tesi di Luca Migliaccio, e successivamente
esteso in un audit metodologico multi-condizione (localizzazione, calibrazione,
robustezza) confluito nel paper *"Beyond a Single Comparison: A Multi-Condition Audit
of Attribution Localization, Calibration, and Robustness in Multi-Label Chest X-Ray
Classification"* (sottomesso a Computerized Medical Imaging and Graphics).

Confronta più architetture (CNN e Vision Transformer) sul dataset NIH ChestX-ray14,
con una pipeline sperimentale completa: preparazione dati, training (baseline e
"avanzato"), ottimizzazione delle soglie, calibrazione delle probabilità,
interpretabilità (Grad-CAM/Eigen-CAM), test di robustezza e analisi degli errori,
anche in configurazione ensemble.

## Dataset

- **NIH ChestX-ray14** (`archive/`) — dataset principale. CSV `Data_Entry_2017.csv` con
  colonne `Image Index` e `Finding Labels` (etichette multiple separate da `|`).
  14 patologie classificate come multi-label (`Atelectasis`, `Cardiomegaly`, `Effusion`,
  `Infiltration`, `Mass`, `Nodule`, `Pneumonia`, `Pneumothorax`, `Consolidation`, `Edema`,
  `Emphysema`, `Fibrosis`, `Pleural_Thickening`, `Hernia`). In alcuni script è presente
  anche una 15ª classe `No Finding` (paziente sano) trattata come categoria a sé. Include
  inoltre il sottoinsieme ufficiale di 984 bounding box patologiche
  (`BBox_List_2017.csv`), usato per la validazione di localizzazione contro ground truth
  reale nel paper CMIG.
- **CheXpert** (`archiveCheXpert/`) — usato per il pre-training di alcuni backbone
  (es. ConvNeXt, EfficientNet-B7) e per script di valutazione dedicati
  (`evalConvNeXtCheXpert.py`).
- Pesi biomedicali esterni: un ConvNeXt-tiny pre-addestrato su CheXpert scaricato da
  Hugging Face (`nicolas-metral/convnext-tiny-chexpert`, via `timm`), e pesi
  **torchxrayvision (XRV)** per DenseNet121/ResNet50 (script con suffisso `-xrv`), che
  richiedono un preprocessing specifico (immagine in scala di grigi, normalizzazione XRV).

## Pipeline / metodologia (verificata nel codice)

1. **Pulizia dati** (`cleaning.py`) — verifica esistenza e integrità di ogni immagine
   (apertura + `img.verify()`), raccoglie statistiche su dimensioni e modalità colore.
2. **Split paziente-level** (`split.py`, `DataLeakage.py`) — split 70/15/15
   (train/val/test) fatto per **PatientID** (estratto dal nome file, non per singola
   immagine), seed fisso `42`, per evitare che immagini dello stesso paziente finiscano
   in split diversi. `DataLeakage.py` verifica l'assenza di overlap tra i tre set.
3. **Training**:
   - Versioni **baseline** (es. `trainDensenet121.py`): 5 epoche, `BCEWithLogitsLoss`
     senza pesi, ottimizzatore Adam, nessun bilanciamento delle classi.
   - Versioni **"Advanced"** (es. `trainAdvancedDensenet121.py`): fino a 20 epoche,
     `BCEWithLogitsLoss` con `pos_weight` per classe (per compensare lo sbilanciamento
     tra patologie rare e frequenti), AdamW con weight decay, mixed precision (AMP),
     gradient clipping, early stopping (pazienza 5).
   - Versioni **"Debiasing"** (es. `trainConvNeXt-Debiasing.py`): aggiungono un
     `CenterCrop` dopo il resize, per ridurre l'effetto di bias legati ai bordi/marker
     dell'immagine radiografica piuttosto che al contenuto clinico. Nell'ablation del
     paper CMIG, questa modifica è stata isolata dagli altri cambi (loss, schedule,
     augmentation) per verificarne l'effetto reale — vedi sezione dedicata più sotto.
   - Modelli Transformer (SwinV2, Rad-DINO) caricati via `transformers`/moduli dedicati,
     con batch size ridotto e gradient accumulation (batch effettivo maggiore), backbone
     congelato per le prime epoche, poi fine-tuning con scheduler coseno.
4. **Valutazione** (`eval*.py`) — inferenza sul test set, ROC-AUC per classe e media,
   `classification_report` a soglia fissa 0.5.
5. **Ottimizzazione soglie** (`threshold*.py`) — per ciascuna classe, grid search della
   soglia (0.05–0.95, step 0.05) che massimizza l'F1-score sul validation set (nella
   pipeline ConvNeXt del paper CMIG, invece, le soglie sono calcolate con l'indice di
   Youden — vedi `evalConvNeXt.py`/`evalConvNeXt-Debiasing.py`); soglie salvate nei file
   `optimized_thresholds*.json`.
6. **Calibrazione** (`calibration*.py`) — reliability diagram per classe
   (`sklearn.calibration.calibration_curve`) ed Expected Calibration Error (ECE).
7. **Interpretabilità** (`grad-cam*.py`, `eigen-cam*.py`, altri script in
   `PythonCode/`) — Grad-CAM/Eigen-CAM (libreria `pytorch_grad_cam`) su una classe
   target, con generazione bilanciata di esempi Veri Positivi/Negativi e Falsi
   Positivi/Negativi (10 per categoria) per l'analisi qualitativa in tesi.
8. **Robustezza** (`robustness*.py`) — stress test applicando rumore gaussiano e blur
   alle immagini di test, misura dell'entropia (media delle entropie binarie per classe,
   non entropia multiclasse) delle predizioni come proxy dell'incertezza del modello
   sotto perturbazione.
9. **Analisi errori ed ensemble** (`erroranalysis*.py`) — analisi dei casi mal
   classificati per singolo modello e in configurazione ensemble (es. DenseNet121 +
   ResNet50 con pesi XRV, ciascuno con soglia ottimizzata individualmente); alcuni script
   esplorano anche combinazioni pesate ottimizzate con Optuna o un meta-modello Random
   Forest sopra le predizioni dei singoli modelli.

## Architetture coperte

DenseNet121, DenseNet169, ResNet50, EfficientNet B0/B3/B7, ConvNeXt (base, via `timm`,
pre-addestrato ImageNet-22k, incluse varianti "Debiasing"), SwinV2 (via `transformers`,
incluse varianti radiology-pretrained), DINOv2, Rad-DINO, oltre a diverse combinazioni
ensemble delle precedenti. Alcuni modelli sono pre-addestrati su dataset biomedicali
(XRV / weights biomedicali, via `torchxrayvision`).

## Struttura del progetto

```
xRayProject/
├── PythonCode/                    # script originali di training/valutazione/analisi
│   ├── train_split.csv            # split paziente per il training (70%)
│   ├── val_split.csv              # split paziente per la validazione (15%)
│   ├── test_split.csv             # split paziente per il test (15%)
│   └── ... (vedi tabella sotto)
│
├── archive/                       # dataset NIH ChestX-ray14 (images_001..images_012,
│                                   #   Data_Entry_2017.csv, BBox_List_2017.csv)
│                                   #   -- NON incluso nel repository, vedi sotto
├── archiveCheXpert/                # dataset CheXpert -- NON incluso nel repository
│
├── checkpoints/                   # pesi dei modelli addestrati e file di supporto
│                                   #   (soglie ottimizzate, error analysis, config
│                                   #   ensemble) -- NON incluso nel repository
│
├── xrv_env/                       # virtual environment Python -- NON incluso
│
└── paper_reproducibility/         # materiali specifici del paper CMIG (vedi sotto)
```

Tutti gli script originali si trovano in `PythonCode/` (struttura non modificata).
Mappa indicativa per categoria funzionale:

| Categoria | Prefisso file | Esempi |
|---|---|---|
| Preparazione/esplorazione dati | vario | `dataset.py`, `datasetXRV.py`, `preprocessing.py`, `cleaning.py`, `split.py`, `DataLeakage.py`, `EsplorazioneCheXpert.py`, `PercentualiPatologie.py`, `CaricamentoPesiBiomedicali.py` |
| Training | `train*.py` | `trainDensenet121.py`, `trainAdvancedDensenet121.py`, `trainConvNeXt.py`, `trainConvNeXt-Debiasing.py`, `trainSwinV2-ChestXRay.py`, `trainRad-Dino.py`, ... |
| Valutazione | `eval*.py` | `evalDensenet121.py`, `evalConvNeXt.py`, `evalConvNeXt-Debiasing.py`, `evalEnsembleConvNeXtSwinV2.py`, ... |
| Ottimizzazione soglie | `threshold*.py` | `thresholdResnet50.py`, `thresholdAdvancedDensenet121(2).py`, ... |
| Analisi errori / ensemble | `erroranalysis*.py` | `erroranalysisDensenet121(2).py`, `erroranalysisEnsembleDensenet121Resnet50-xrv.py`, `erroranalysisEnsembleOptunaDensenet121Resnet50-xrv.py`, ... |
| Calibrazione | `calibration*.py` | `calibrationConvNeXt.py`, `calibrationConvNeXt-Debiasing.py`, `calibration-robustnessConvNeXt-Debiasing.py` |
| Interpretabilità | `grad-cam*.py`, `eigen-cam*.py` | `grad-camConvNeXt.py`, `grad-camConvNeXt-DebiasingFinale.py`, `eigen-camSwinV2.py`, `ActivationMaximationConvNeXt.py`, `DifferenceMapConvNeXt.py`, `GuidedActivationConvNeXt.py` |
| Robustezza | `robustness*.py` | `robustnessConvNeXt.py`, `robustnessConvNeXtDebiasing2.py` |
| Grafici riassuntivi | vario | `GenerazioneGraficoRobustezza.py`, `GenerazioneIstogrammi.py` |
| Script esplorativi/prove | `prova*.py` | `prova.py` (vuoto), `prova1.py`, `provaEnsemble.py`, `analysisSwinV2.py` |

## ⚠️ Percorsi hardcoded

Molti script contengono percorsi assoluti scritti direttamente nel codice, del tipo:
```python
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
train_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv"
```
Per eseguire gli script su un'altra macchina (o dopo aver clonato il repo altrove),
questi percorsi vanno aggiornati manualmente — non sono relativi al repository.
Alcuni script più recenti (tutti quelli in `paper_reproducibility/`, vedi sotto) usano
lo stesso pattern ma con le variabili di configurazione raccolte in cima al file,
rendendo l'aggiornamento più semplice; nessuno script del progetto usa ancora
`BASE_DIR = os.path.dirname(os.path.abspath(__file__))` in modo sistematico — verifica
sempre le variabili di percorso prima di lanciare uno script su una macchina diversa da
quella originale.

## Dati e file esclusi dal repository

Questo repository contiene **solo il codice**. Sono esclusi (vedi `.gitignore`):
- I dataset (`archive/`, `archiveCheXpert/`) — diverse decine di GB
- Il virtual environment Python (`xrv_env/`)
- Checkpoint / pesi dei modelli addestrati
- Output di analisi generati (grafici, immagini Grad-CAM, report)

Sono invece **inclusi** (per riproducibilità): `train_split.csv`, `val_split.csv`,
`test_split.csv` (split esatto usato negli esperimenti) e i file
`optimized_thresholds*.json` (soglie per classe usate nelle fasi successive alla
valutazione).

Scarica il dataset NIH ChestX-ray14 separatamente e posizionalo sotto `archive/`
rispettando la struttura a cartelle `images_001/` .. `images_012/` attesa da tutti gli
script. Analogamente per CheXpert sotto `archiveCheXpert/`, dove usato.

## Ambiente e versioni software

Python 3.12.3, testato con:

```
torch==2.12.0+cu130
torchvision==0.27.0+cu130
timm==1.0.27
scikit-learn==1.8.0
scipy==1.17.1
pandas==3.0.3
numpy==2.4.6
scikit-image==0.26.0
grad-cam
```

Per ricreare l'ambiente:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Nota**: `requirements.txt` include alcune dipendenze CUDA/NVIDIA specifiche
> dell'ambiente originale (GPU e driver della VM su cui è stato sviluppato il progetto).
> Su una macchina con GPU/driver diversi, o senza GPU, potrebbe essere necessario
> adattare le versioni di `torch` e dei pacchetti `nvidia-*`.

Se `requirements.txt` non è ancora presente/aggiornato nella cartella, rigeneralo dalla
macchina con l'ambiente `xrv_env` attivo:

```bash
source xrv_env/bin/activate
pip freeze > requirements.txt
```

così chiunque cloni il repo può ricreare l'ambiente senza il venv originale.

## Come usare gli script

Tutti gli script di `PythonCode/` seguono lo stesso pattern: **configurazione tramite
variabili in cima al file** (path del dataset, dei checkpoint, iperparametri), non
tramite argomenti da riga di comando (vedi anche "⚠️ Percorsi hardcoded" sopra). Prima
di lanciare uno script, aprilo e verifica/aggiorna le variabili di percorso in testa al
file (tipicamente `ROOT_DIR`, `CHECKPOINT_DIR`, `TRAIN_CSV`/`VAL_CSV`/`TEST_CSV`), poi
eseguilo direttamente:

```bash
cd PythonCode
python <nome_script>.py
```

L'ordine seguente è dedotto dalla logica della pipeline e dal codice dei singoli
script; non è stato verificato con un run end-to-end completo di tutte le
combinazioni architettura/variante, quindi in caso di errore controlla prima le
variabili di configurazione in cima allo script che stai lanciando.

**1. Preparazione dati** (una tantum, prima di tutto il resto)
```bash
python cleaning.py           # valida integrità e formato del dataset scaricato
python split.py              # genera train_split.csv / val_split.csv / test_split.csv
                              # a livello paziente (70/15/15, seed 42)
python DataLeakage.py        # verifica assenza di overlap paziente fra gli split
python PercentualiPatologie.py   # statistiche di distribuzione delle classi
python EsplorazioneCheXpert.py   # esplorazione dataset CheXpert, se usato
python CaricamentoPesiBiomedicali.py   # scarica/verifica i pesi pre-addestrati XRV
```

**2. Training** (un modello per script; ogni script salva il proprio checkpoint in
`checkpoints/`)
```bash
python trainConvNeXt.py              # ConvNeXt, configurazione pre-intervento
python trainConvNeXt-Debiasing.py    # ConvNeXt, configurazione post-intervento (CenterCrop)
python trainDensenet121.py           # DenseNet-121, versione baseline (5 epoche)
python trainAdvancedDensenet121.py   # DenseNet-121, versione avanzata (pos_weight, AMP, early stopping)
python trainSwinV2-ChestXRay.py      # SwinV2, pesi radiology-pretrained
python trainRad-Dino.py              # Rad-DINO
# ... e gli equivalenti train*.py per le altre architetture elencate sopra
```

**3. Ottimizzazione soglie e valutazione**
```bash
python thresholdResnet50.py            # grid search soglia F1-ottimale per singola architettura
python thresholdAdvancedDensenet121\(2\).py
python evalConvNeXt.py                 # metriche globali + soglie di Youden (pre-intervento)
python evalConvNeXt-Debiasing.py       # idem, post-intervento
python evalDensenet121.py
python evalEnsembleConvNeXtSwinV2.py   # metriche dell'ensemble a inferenza
```

**4. Analisi degli errori ed ensemble**
```bash
python erroranalysisDensenet121\(2\).py
python erroranalysisEnsembleDensenet121Resnet50-xrv.py
python erroranalysisEnsembleOptunaDensenet121Resnet50-xrv.py   # combinazione pesata via Optuna
```

**5. Interpretabilità (Grad-CAM / Eigen-CAM)**
```bash
python grad-camConvNeXt.py                  # mappe Grad-CAM, modello pre-intervento
python grad-camConvNeXt-DebiasingFinale.py  # report visivi su campione di 1.000 immagini,
                                             #   modello post-intervento
python eigen-camSwinV2.py
python ActivationMaximationConvNeXt.py      # visualizzazione per massimizzazione di attivazione
python DifferenceMapConvNeXt.py             # mappa differenza fra attivazioni pre/post
python GuidedActivationConvNeXt.py
```

**6. Calibrazione**
```bash
python calibrationConvNeXt.py                    # ECE, modello pre-intervento
python calibrationConvNeXt-Debiasing.py          # ECE, modello post-intervento (10 bin uniformi)
python calibration-robustnessConvNeXt-Debiasing.py   # calibrazione sotto corruzione
```

**7. Robustezza**
```bash
python robustnessConvNeXt.py             # rumore/blur, modello pre-intervento
python robustnessConvNeXtDebiasing2.py   # idem, modello post-intervento
```

**8. Grafici riassuntivi**
```bash
python GenerazioneGraficoRobustezza.py
python GenerazioneIstogrammi.py
```

Gli script con prefisso `prova*.py` (`prova.py` — vuoto, `prova1.py`, `provaEnsemble.py`,
`analysisSwinV2.py`) sono script esplorativi/di prova usati durante lo sviluppo, non
parte della pipeline finale: consultali per riferimento ma non fanno parte del percorso
di riproduzione standard.

---

## Materiali di riproducibilità per il paper CMIG

`paper_reproducibility/` contiene tutto il necessario per riprodurre il contributo
metodologico centrale del paper: un ablation controllato a sette condizioni e tre
seed dell'intervento motivato dall'explainability (isolando la modifica "Debiasing"
menzionata sopra — CenterCrop — dagli altri cambi confusi insieme nella pipeline
originale: loss function, training schedule, augmentation), validato sia contro un
proxy body-silhouette sia contro le bounding box patologiche ufficiali NIH, insieme a
un risultato di calibrazione emerso indipendentemente.

```
paper_reproducibility/
├── requirements.txt
├── ablation/
│   └── ablation_train_and_eval.py
├── localization_analysis/
│   ├── quantitative_localization_debiasing.py
│   ├── recompute_localization_fixed_mask.py
│   ├── bbox_pointing_game_localization.py
│   ├── analyze_ablation_results.py
│   ├── analyze_bbox_results.py
│   └── visual_qc_c0_vs_c6.py
├── calibration_analysis/
│   ├── full_calibration_reanalysis.py
│   ├── diagnose_ece_discrepancy.py
│   ├── recompute_table7_per_class_ece.py
│   └── regenerate_reliability_diagram.py
├── inventory_checkpoints.py
├── build_final_checkpoint_manifest.py
└── results/
    ├── SUMMARY_global_metrics.csv
    ├── SUMMARY_localization_vs_baseline.csv
    ├── SUMMARY_bbox_localization.csv
    ├── SUMMARY_bbox_statistical_analysis.csv
    ├── SUMMARY_calibration_full.csv
    ├── SUMMARY_calibration_per_class.csv
    ├── table7_corrected.csv
    ├── checkpoint_inventory_FULL.csv
    ├── checkpoint_hashes.csv
    ├── diagnostic_original_checkpoint_predictions.csv
    ├── test_predictions_<condizione>_seed<seed>.csv
    ├── localization_<condizione>_seed<seed>_FIXEDMASK.csv
    ├── bbox_localization_<condizione>_seed<seed>.csv
    ├── figures/
    │   └── fig7_reliability_diagram_CORRECTED_Effusion.png
    ├── qc/
    │   ├── mask_comparison/mask_old_vs_fixed.png
    │   └── visual_qc/qc_grid_seed42.png
    └── patient_splits/
        ├── train_split.csv
        ├── val_split.csv
        └── test_split.csv
```

### Sequenza di riproduzione

```bash
cd paper_reproducibility

# 1. Allena e valuta le 7 condizioni dell'ablation x 3 seed (Table 4-5)
#    -- passo costoso (ore/giorni); puoi lanciarlo una condizione alla volta
python ablation/ablation_train_and_eval.py --conditions C0_baseline
python ablation/ablation_train_and_eval.py --conditions C1_crop_only
# ... ripeti per C2_bce_only, C3_schedule_only, C4_noaug_only, C5_crop_bce, C6_full_combined

# 2. Analisi statistica dell'ablation di localizzazione (Table 5):
#    bootstrap patient-only e gerarchico (seed poi paziente)
python localization_analysis/analyze_ablation_results.py

# 3. Correggi l'artefatto della maschera body-silhouette (esclusione della spalla,
#    trovato tramite controllo visivo -- vedi visual_qc_c0_vs_c6.py) e ricalcola
python localization_analysis/recompute_localization_fixed_mask.py

# 4. Validazione con bounding box vere (Table 6) -- solo inferenza, nessun training
python localization_analysis/bbox_pointing_game_localization.py
python localization_analysis/analyze_bbox_results.py

# 5. Ri-analisi della calibrazione su tutte le condizioni dell'ablation (Table 8)
#    -- pura ri-analisi dei CSV già salvati, nessuna inferenza nuova
python calibration_analysis/full_calibration_reanalysis.py

# 6. Diagnosi della discrepanza ECE e correzione di Table 7 / Figure 7
python calibration_analysis/diagnose_ece_discrepancy.py
python calibration_analysis/recompute_table7_per_class_ece.py
python calibration_analysis/regenerate_reliability_diagram.py

# 7. Manifest dei checkpoint (inventario completo + whitelist curata con hash)
python inventory_checkpoints.py
python build_final_checkpoint_manifest.py
```

Il docstring di ogni script documenta input, output, tempo di esecuzione atteso ed
eventuali avvertenze metodologiche (scelte di preprocessing, assunzioni statistiche).
Leggilo prima di lanciare lo script.

### Cosa ha stabilito ciascuna analisi

- **Localizzazione body-silhouette (Table 5)**: un confronto singolo e non
  controllato prima/dopo l'intervento suggeriva inizialmente un miglioramento forte
  ($p=1.4\times10^{-28}$). Una volta corretto l'artefatto di esclusione della spalla
  e valutato con bootstrap gerarchico (seed poi paziente) su 3 seed, nessuna
  condizione mostra un effetto distinguibile dal rumore di training.
- **Localizzazione con bounding box vere (Table 6)**: ristretta alle 131 immagini
  uniche di test con annotazioni ufficiali NIH (dopo la correzione di un bug di
  contaminazione train/test), la pointing-game accuracy non mostra un miglioramento
  statisticamente robusto; un piccolo calo della massa di attivazione dentro il
  riquadro reale della lesione è l'unico intervallo che esclude lo zero, un
  risultato esplorativo non corretto per molteplicità.
- **Calibrazione (Table 7-8)**: l'Asymmetric Loss è associata a un Expected
  Calibration Error circa 29 volte più alto della binary cross-entropy, in modo
  concordante fra ECE, Brier score e negative log-likelihood. Una discrepanza
  iniziale fra questo pattern interno all'ablation e una stima calcolata
  separatamente su un'unica istanza (0.1015 vs ~0.01) è stata ricondotta, tramite
  `diagnose_ece_discrepancy.py`, a una pipeline di valutazione precedente e
  superata, non al modello stesso.

### Checkpoint

`paper_reproducibility/results/checkpoint_hashes.csv` elenca gli hash SHA-256 di
ogni checkpoint citato nel paper (26 disponibili: 21 dell'ablation + 5
single-backbone/riferimento per Table 4/7-8, più 2 documentati esplicitamente come
non disponibili). Due configurazioni di Table 4 (ResNet-50 e DenseNet-121, entrambe
su pre-training ImageNet puro) sono state allenate localmente dallo studente prima
del passaggio al server GPU usato per tutto il resto del progetto; i relativi
checkpoint non sono mai stati caricati e non sono recuperabili. Per quelle due righe
sono verificabili solo le metriche già riportate nel paper, non i pesi del modello.

I checkpoint veri e propri non sono tracciati in git (vedi `.gitignore`); vanno
ospitati separatamente (es. Zenodo).

## Autore

Luca Migliaccio — tesi seguita da Antonio Esposito, Università degli Studi della
Campania "Luigi Vanvitelli".

---
---

# English version

# Chest X-Ray Classification — Migliaccio Thesis

## Description

Multi-label chest X-ray classification project using deep learning, developed as part
of Luca Migliaccio's thesis, and later extended into a multi-condition methodological
audit (localization, calibration, robustness) that fed into the paper *"Beyond a
Single Comparison: A Multi-Condition Audit of Attribution Localization, Calibration,
and Robustness in Multi-Label Chest X-Ray Classification"* (submitted to
Computerized Medical Imaging and Graphics).

Compares multiple architectures (CNNs and Vision Transformers) on the NIH
ChestX-ray14 dataset, with a complete experimental pipeline: data preparation,
training (baseline and "advanced"), threshold optimization, probability calibration,
interpretability (Grad-CAM/Eigen-CAM), robustness testing, and error analysis,
including ensemble configurations.

## Dataset

- **NIH ChestX-ray14** (`archive/`) — main dataset. `Data_Entry_2017.csv` with columns
  `Image Index` and `Finding Labels` (multiple labels separated by `|`). 14
  pathologies classified as multi-label (`Atelectasis`, `Cardiomegaly`, `Effusion`,
  `Infiltration`, `Mass`, `Nodule`, `Pneumonia`, `Pneumothorax`, `Consolidation`,
  `Edema`, `Emphysema`, `Fibrosis`, `Pleural_Thickening`, `Hernia`). Some scripts also
  include a 15th class, `No Finding` (healthy patient), treated as its own category.
  Also includes the official subset of 984 pathology bounding boxes
  (`BBox_List_2017.csv`), used for localization validation against real ground truth
  in the CMIG paper.
- **CheXpert** (`archiveCheXpert/`) — used to pretrain some backbones (e.g. ConvNeXt,
  EfficientNet-B7) and for dedicated evaluation scripts (`evalConvNeXtCheXpert.py`).
- External biomedical weights: a ConvNeXt-tiny pretrained on CheXpert and downloaded
  from Hugging Face (`nicolas-metral/convnext-tiny-chexpert`, via `timm`), and
  **torchxrayvision (XRV)** weights for DenseNet121/ResNet50 (scripts with the `-xrv`
  suffix), which require specific preprocessing (grayscale image, XRV normalization).

## Pipeline / methodology (verified in code)

1. **Data cleaning** (`cleaning.py`) — checks the existence and integrity of every
   image (open + `img.verify()`), collects statistics on size and color mode.
2. **Patient-level split** (`split.py`, `DataLeakage.py`) — 70/15/15 split
   (train/val/test) done by **PatientID** (extracted from the filename, not per
   individual image), fixed seed `42`, to prevent images of the same patient from
   ending up in different splits. `DataLeakage.py` verifies there is no overlap
   across the three sets.
3. **Training**:
   - **Baseline** versions (e.g. `trainDensenet121.py`): 5 epochs, unweighted
     `BCEWithLogitsLoss`, Adam optimizer, no class balancing.
   - **"Advanced"** versions (e.g. `trainAdvancedDensenet121.py`): up to 20 epochs,
     `BCEWithLogitsLoss` with per-class `pos_weight` (to compensate for the imbalance
     between rare and frequent pathologies), AdamW with weight decay, mixed precision
     (AMP), gradient clipping, early stopping (patience 5).
   - **"Debiasing"** versions (e.g. `trainConvNeXt-Debiasing.py`): add a `CenterCrop`
     after the resize, to reduce the effect of biases tied to image borders/markers
     rather than clinical content. In the CMIG paper's ablation, this change was
     isolated from the other confounded changes (loss, schedule, augmentation) to
     verify its real effect — see the dedicated section below.
   - Transformer models (SwinV2, Rad-DINO) loaded via `transformers`/dedicated
     modules, with reduced batch size and gradient accumulation (larger effective
     batch), frozen backbone for the first epochs, then fine-tuning with a cosine
     scheduler.
4. **Evaluation** (`eval*.py`) — inference on the test set, per-class and mean
   ROC-AUC, `classification_report` at a fixed 0.5 threshold.
5. **Threshold optimization** (`threshold*.py`) — for each class, a grid search over
   the threshold (0.05–0.95, step 0.05) that maximizes F1-score on the validation set
   (in the CMIG paper's ConvNeXt pipeline, thresholds are instead computed via the
   Youden index — see `evalConvNeXt.py`/`evalConvNeXt-Debiasing.py`); thresholds
   saved to `optimized_thresholds*.json` files.
6. **Calibration** (`calibration*.py`) — per-class reliability diagram
   (`sklearn.calibration.calibration_curve`) and Expected Calibration Error (ECE).
7. **Interpretability** (`grad-cam*.py`, `eigen-cam*.py`, other scripts in
   `PythonCode/`) — Grad-CAM/Eigen-CAM (`pytorch_grad_cam` library) on a target
   class, with balanced generation of True Positive/Negative and False
   Positive/Negative examples (10 per category) for the qualitative analysis in the
   thesis.
8. **Robustness** (`robustness*.py`) — stress test applying Gaussian noise and blur
   to test images, measuring entropy (mean of per-class binary entropies, not
   multiclass entropy) of the predictions as a proxy for model uncertainty under
   perturbation.
9. **Error and ensemble analysis** (`erroranalysis*.py`) — analysis of misclassified
   cases for individual models and in ensemble configuration (e.g. DenseNet121 +
   ResNet50 with XRV weights, each with an individually optimized threshold); some
   scripts also explore weighted combinations optimized with Optuna or a Random
   Forest meta-model on top of individual models' predictions.

## Architectures covered

DenseNet121, DenseNet169, ResNet50, EfficientNet B0/B3/B7, ConvNeXt (base, via `timm`,
ImageNet-22k pretrained, including "Debiasing" variants), SwinV2 (via `transformers`,
including radiology-pretrained variants), DINOv2, Rad-DINO, plus various ensemble
combinations of the above. Some models are pretrained on biomedical datasets (XRV /
biomedical weights, via `torchxrayvision`).

## Project structure

```
xRayProject/
├── PythonCode/                    # original training/evaluation/analysis scripts
│   ├── train_split.csv            # patient-level training split (70%)
│   ├── val_split.csv              # patient-level validation split (15%)
│   ├── test_split.csv             # patient-level test split (15%)
│   └── ... (see table below)
│
├── archive/                       # NIH ChestX-ray14 dataset (images_001..images_012,
│                                   #   Data_Entry_2017.csv, BBox_List_2017.csv)
│                                   #   -- NOT included in this repository, see below
├── archiveCheXpert/                # CheXpert dataset -- NOT included in this repository
│
├── checkpoints/                   # trained model weights and supporting files
│                                   #   (optimized thresholds, error analyses, ensemble
│                                   #   configs) -- NOT included in this repository
│
├── xrv_env/                       # Python virtual environment -- NOT included
│
└── paper_reproducibility/         # materials specific to the CMIG paper (see below)
```

All original scripts are located in `PythonCode/` (unmodified structure). Indicative
map by functional category:

| Category | File prefix | Examples |
|---|---|---|
| Data preparation/exploration | various | `dataset.py`, `datasetXRV.py`, `preprocessing.py`, `cleaning.py`, `split.py`, `DataLeakage.py`, `EsplorazioneCheXpert.py`, `PercentualiPatologie.py`, `CaricamentoPesiBiomedicali.py` |
| Training | `train*.py` | `trainDensenet121.py`, `trainAdvancedDensenet121.py`, `trainConvNeXt.py`, `trainConvNeXt-Debiasing.py`, `trainSwinV2-ChestXRay.py`, `trainRad-Dino.py`, ... |
| Evaluation | `eval*.py` | `evalDensenet121.py`, `evalConvNeXt.py`, `evalConvNeXt-Debiasing.py`, `evalEnsembleConvNeXtSwinV2.py`, ... |
| Threshold optimization | `threshold*.py` | `thresholdResnet50.py`, `thresholdAdvancedDensenet121(2).py`, ... |
| Error analysis / ensemble | `erroranalysis*.py` | `erroranalysisDensenet121(2).py`, `erroranalysisEnsembleDensenet121Resnet50-xrv.py`, `erroranalysisEnsembleOptunaDensenet121Resnet50-xrv.py`, ... |
| Calibration | `calibration*.py` | `calibrationConvNeXt.py`, `calibrationConvNeXt-Debiasing.py`, `calibration-robustnessConvNeXt-Debiasing.py` |
| Interpretability | `grad-cam*.py`, `eigen-cam*.py` | `grad-camConvNeXt.py`, `grad-camConvNeXt-DebiasingFinale.py`, `eigen-camSwinV2.py`, `ActivationMaximationConvNeXt.py`, `DifferenceMapConvNeXt.py`, `GuidedActivationConvNeXt.py` |
| Robustness | `robustness*.py` | `robustnessConvNeXt.py`, `robustnessConvNeXtDebiasing2.py` |
| Summary charts | various | `GenerazioneGraficoRobustezza.py`, `GenerazioneIstogrammi.py` |
| Exploratory/scratch scripts | `prova*.py` | `prova.py` (empty), `prova1.py`, `provaEnsemble.py`, `analysisSwinV2.py` |

## ⚠️ Hardcoded paths

Many scripts contain absolute paths written directly in the code, such as:
```python
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"
train_csv = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode/train_split.csv"
```
To run these scripts on a different machine (or after cloning the repository
elsewhere), these paths must be updated manually — they are not relative to the
repository. Some more recent scripts (all of those in `paper_reproducibility/`, see
below) follow the same pattern but with configuration variables collected at the top
of the file, making updates easier; no script in the project systematically uses
`BASE_DIR = os.path.dirname(os.path.abspath(__file__))` — always check the path
variables before running a script on a machine other than the original one.

## Data and files excluded from the repository

This repository contains **code only**. Excluded (see `.gitignore`):
- The datasets (`archive/`, `archiveCheXpert/`) — tens of GB
- The Python virtual environment (`xrv_env/`)
- Trained model checkpoints / weights
- Generated analysis outputs (charts, Grad-CAM images, reports)

**Included** instead (for reproducibility): `train_split.csv`, `val_split.csv`,
`test_split.csv` (the exact split used in the experiments) and the
`optimized_thresholds*.json` files (per-class thresholds used in the stages
downstream of evaluation).

Download the NIH ChestX-ray14 dataset separately and place it under `archive/`
following the `images_001/` .. `images_012/` folder structure expected by all
scripts. Likewise for CheXpert under `archiveCheXpert/`, where used.

## Environment and software versions

Python 3.12.3, tested with:

```
torch==2.12.0+cu130
torchvision==0.27.0+cu130
timm==1.0.27
scikit-learn==1.8.0
scipy==1.17.1
pandas==3.0.3
numpy==2.4.6
scikit-image==0.26.0
grad-cam
```

To recreate the environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note**: `requirements.txt` includes some CUDA/NVIDIA dependencies specific to the
> original environment (GPU and drivers of the VM the project was developed on). On a
> machine with different GPU/drivers, or without a GPU, you may need to adapt the
> `torch` and `nvidia-*` package versions.

If `requirements.txt` is not yet present/up to date in the folder, regenerate it from
the machine with the `xrv_env` environment active:

```bash
source xrv_env/bin/activate
pip freeze > requirements.txt
```

so that anyone cloning the repository can recreate the environment without the
original venv.

## How to use the scripts

All scripts in `PythonCode/` follow the same pattern: **configuration via variables
at the top of the file** (dataset path, checkpoint path, hyperparameters), not
command-line arguments (see also "⚠️ Hardcoded paths" above). Before running a
script, open it and check/update the path variables at the top of the file
(typically `ROOT_DIR`, `CHECKPOINT_DIR`, `TRAIN_CSV`/`VAL_CSV`/`TEST_CSV`), then run
it directly:

```bash
cd PythonCode
python <script_name>.py
```

The order below is inferred from the pipeline's logic and from the individual
scripts' code; it has not been verified with a complete end-to-end run across every
architecture/variant combination, so if you hit an error, check the configuration
variables at the top of the script you are running first.

**1. Data preparation** (one-time, before everything else)
```bash
python cleaning.py           # validates integrity and format of the downloaded dataset
python split.py              # generates train_split.csv / val_split.csv / test_split.csv
                              # at the patient level (70/15/15, seed 42)
python DataLeakage.py        # verifies no patient overlap across splits
python PercentualiPatologie.py   # class-distribution statistics
python EsplorazioneCheXpert.py   # CheXpert dataset exploration, if used
python CaricamentoPesiBiomedicali.py   # downloads/verifies XRV pretrained weights
```

**2. Training** (one model per script; each script saves its own checkpoint to
`checkpoints/`)
```bash
python trainConvNeXt.py              # ConvNeXt, pre-intervention configuration
python trainConvNeXt-Debiasing.py    # ConvNeXt, post-intervention configuration (CenterCrop)
python trainDensenet121.py           # DenseNet-121, baseline version (5 epochs)
python trainAdvancedDensenet121.py   # DenseNet-121, advanced version (pos_weight, AMP, early stopping)
python trainSwinV2-ChestXRay.py      # SwinV2, radiology-pretrained weights
python trainRad-Dino.py              # Rad-DINO
# ... and the equivalent train*.py scripts for the other architectures listed above
```

**3. Threshold optimization and evaluation**
```bash
python thresholdResnet50.py            # F1-optimal threshold grid search for a single architecture
python thresholdAdvancedDensenet121\(2\).py
python evalConvNeXt.py                 # global metrics + Youden thresholds (pre-intervention)
python evalConvNeXt-Debiasing.py       # same, post-intervention
python evalDensenet121.py
python evalEnsembleConvNeXtSwinV2.py   # ensemble metrics at inference time
```

**4. Error and ensemble analysis**
```bash
python erroranalysisDensenet121\(2\).py
python erroranalysisEnsembleDensenet121Resnet50-xrv.py
python erroranalysisEnsembleOptunaDensenet121Resnet50-xrv.py   # Optuna-optimized weighted combination
```

**5. Interpretability (Grad-CAM / Eigen-CAM)**
```bash
python grad-camConvNeXt.py                  # Grad-CAM maps, pre-intervention model
python grad-camConvNeXt-DebiasingFinale.py  # visual reports on a 1,000-image sample,
                                             #   post-intervention model
python eigen-camSwinV2.py
python ActivationMaximationConvNeXt.py      # activation-maximization visualization
python DifferenceMapConvNeXt.py             # difference map between pre/post activations
python GuidedActivationConvNeXt.py
```

**6. Calibration**
```bash
python calibrationConvNeXt.py                    # ECE, pre-intervention model
python calibrationConvNeXt-Debiasing.py          # ECE, post-intervention model (10 uniform bins)
python calibration-robustnessConvNeXt-Debiasing.py   # calibration under corruption
```

**7. Robustness**
```bash
python robustnessConvNeXt.py             # noise/blur, pre-intervention model
python robustnessConvNeXtDebiasing2.py   # same, post-intervention model
```

**8. Summary charts**
```bash
python GenerazioneGraficoRobustezza.py
python GenerazioneIstogrammi.py
```

Scripts prefixed `prova*.py` (`prova.py` — empty, `prova1.py`, `provaEnsemble.py`,
`analysisSwinV2.py`) are exploratory/scratch scripts used during development, not
part of the final pipeline: consult them for reference, but they are not part of the
standard reproduction path.

---

## Reproducibility materials for the CMIG paper

`paper_reproducibility/` contains everything needed to reproduce the paper's central
methodological contribution: a controlled, seven-condition, three-seed ablation of
the explainability-motivated intervention (isolating the "Debiasing" change mentioned
above — CenterCrop — from the other changes confounded together in the original
pipeline: loss function, training schedule, augmentation), validated against both a
body-silhouette proxy and NIH's official pathology bounding boxes, alongside an
independently surfaced calibration finding.

```
paper_reproducibility/
├── requirements.txt
├── ablation/
│   └── ablation_train_and_eval.py
├── localization_analysis/
│   ├── quantitative_localization_debiasing.py
│   ├── recompute_localization_fixed_mask.py
│   ├── bbox_pointing_game_localization.py
│   ├── analyze_ablation_results.py
│   ├── analyze_bbox_results.py
│   └── visual_qc_c0_vs_c6.py
├── calibration_analysis/
│   ├── full_calibration_reanalysis.py
│   ├── diagnose_ece_discrepancy.py
│   ├── recompute_table7_per_class_ece.py
│   └── regenerate_reliability_diagram.py
├── inventory_checkpoints.py
├── build_final_checkpoint_manifest.py
└── results/
    ├── SUMMARY_global_metrics.csv
    ├── SUMMARY_localization_vs_baseline.csv
    ├── SUMMARY_bbox_localization.csv
    ├── SUMMARY_bbox_statistical_analysis.csv
    ├── SUMMARY_calibration_full.csv
    ├── SUMMARY_calibration_per_class.csv
    ├── table7_corrected.csv
    ├── checkpoint_inventory_FULL.csv
    ├── checkpoint_hashes.csv
    ├── diagnostic_original_checkpoint_predictions.csv
    ├── test_predictions_<condition>_seed<seed>.csv
    ├── localization_<condition>_seed<seed>_FIXEDMASK.csv
    ├── bbox_localization_<condition>_seed<seed>.csv
    ├── figures/
    │   └── fig7_reliability_diagram_CORRECTED_Effusion.png
    ├── qc/
    │   ├── mask_comparison/mask_old_vs_fixed.png
    │   └── visual_qc/qc_grid_seed42.png
    └── patient_splits/
        ├── train_split.csv
        ├── val_split.csv
        └── test_split.csv
```

### Reproduction sequence

```bash
cd paper_reproducibility

# 1. Train and evaluate all 7 ablation conditions x 3 seeds (Tables 4-5)
#    -- the expensive step (hours to days); can be run condition by condition
python ablation/ablation_train_and_eval.py --conditions C0_baseline
python ablation/ablation_train_and_eval.py --conditions C1_crop_only
# ... repeat for C2_bce_only, C3_schedule_only, C4_noaug_only, C5_crop_bce, C6_full_combined

# 2. Statistical analysis of the localization ablation (Table 5):
#    patient-only and hierarchical (seed-then-patient) bootstrap
python localization_analysis/analyze_ablation_results.py

# 3. Correct the body-silhouette mask's shoulder-exclusion artifact
#    (found via visual audit -- see visual_qc_c0_vs_c6.py) and recompute
python localization_analysis/recompute_localization_fixed_mask.py

# 4. Ground-truth bounding-box validation (Table 6) -- inference only, no training
python localization_analysis/bbox_pointing_game_localization.py
python localization_analysis/analyze_bbox_results.py

# 5. Calibration reanalysis across all ablation conditions (Table 8)
#    -- pure reanalysis of already-saved CSVs, no new inference
python calibration_analysis/full_calibration_reanalysis.py

# 6. ECE discrepancy diagnosis and Table 7 / Figure 7 correction
python calibration_analysis/diagnose_ece_discrepancy.py
python calibration_analysis/recompute_table7_per_class_ece.py
python calibration_analysis/regenerate_reliability_diagram.py

# 7. Checkpoint manifest (full inventory + curated whitelist with hashes)
python inventory_checkpoints.py
python build_final_checkpoint_manifest.py
```

Each script's docstring documents its inputs, outputs, expected runtime, and any
methodological caveats (preprocessing choices, statistical assumptions). Read it
before running the script.

### What each analysis established

- **Body-silhouette localization (Table 5)**: a single, uncontrolled before/after
  comparison of the intervention initially suggested a strong improvement
  ($p=1.4\times10^{-28}$). Once corrected for a shoulder-exclusion mask artifact and
  evaluated with a hierarchical (seed-then-patient) bootstrap across 3 seeds, no
  condition's effect is distinguishable from training noise.
- **Ground-truth bounding-box localization (Table 6)**: restricted to the 131 unique
  test-split images with official NIH annotations (after fixing an initial
  train/test contamination bug), pointing-game accuracy shows no statistically
  robust improvement; a small decrease in activation mass inside the true lesion box
  is the only interval excluding zero, an exploratory, uncorrected-for-multiplicity
  result.
- **Calibration (Tables 7-8)**: Asymmetric Loss is associated with roughly 29-fold
  higher Expected Calibration Error than binary cross-entropy, concordant across
  ECE, Brier score, and negative log-likelihood. An initial discrepancy between this
  ablation-internal pattern and a separately-computed single-instance estimate
  (0.1015 vs. ~0.01) was traced, via `diagnose_ece_discrepancy.py`, to an earlier,
  superseded evaluation pipeline rather than to the model itself.

### Checkpoints

`paper_reproducibility/results/checkpoint_hashes.csv` lists SHA-256 hashes for every
checkpoint referenced in the paper (26 available: 21 ablation checkpoints + 5
single-backbone/reference checkpoints for Tables 4/7-8, plus 2 explicitly documented
as unavailable). Two Table 4 configurations (ResNet-50 and DenseNet-121, both on
plain ImageNet pretraining) were trained locally by the student before migrating the
project to the GPU server used for everything else in this repository; their
checkpoints were never uploaded and are not recoverable. Only the metrics already
reported in the paper are verifiable for those two rows, not the underlying model
weights.

Trained model checkpoints themselves are not tracked in git (see `.gitignore`); host
them separately (e.g., Zenodo).

## Author

Luca Migliaccio — thesis supervised by Antonio Esposito, Università degli Studi
della Campania "Luigi Vanvitelli".
