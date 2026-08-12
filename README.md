# Chest X-Ray Classification — Tesi Migliaccio

## Descrizione

Progetto di classificazione multi-label di radiografie del torace (chest X-ray) tramite
deep learning, sviluppato nell'ambito della tesi di Luca Migliaccio. Confronta più
architetture (CNN e Vision Transformer) sul dataset NIH ChestX-ray14, con una pipeline
sperimentale completa: preparazione dati, training (baseline e "avanzato"), ottimizzazione
delle soglie, calibrazione delle probabilità, interpretabilità (Grad-CAM/eigen-CAM),
test di robustezza e analisi degli errori, anche in configurazione ensemble.

## Dataset

- **NIH ChestX-ray14** (`archive/`) — dataset principale. CSV `Data_Entry_2017.csv` con
  colonne `Image Index` e `Finding Labels` (etichette multiple separate da `|`).
  14 patologie classificate come multi-label (`Atelectasis`, `Cardiomegaly`, `Effusion`,
  `Infiltration`, `Mass`, `Nodule`, `Pneumonia`, `Pneumothorax`, `Consolidation`, `Edema`,
  `Emphysema`, `Fibrosis`, `Pleural_Thickening`, `Hernia`). In alcuni script è presente
  anche una 15ª classe `No Finding` (paziente sano) trattata come categoria a sé.
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
     dell'immagine radiografica piuttosto che al contenuto clinico.
   - Modelli Transformer (SwinV2, Rad-DINO) caricati via `transformers`/moduli dedicati,
     con batch size ridotto e gradient accumulation (batch effettivo maggiore), backbone
     congelato per le prime epoche, poi fine-tuning con scheduler coseno.
4. **Valutazione** (`eval*.py`) — inferenza sul test set, ROC-AUC per classe e media,
   `classification_report` a soglia fissa 0.5.
5. **Ottimizzazione soglie** (`threshold*.py`) — per ciascuna classe, grid search della
   soglia (0.05–0.95, step 0.05) che massimizza l'F1-score sul validation set; soglie
   salvate nei file `optimized_thresholds*.json`.
6. **Calibrazione** (`calibration*.py`) — reliability diagram per classe
   (`sklearn.calibration.calibration_curve`) ed Expected Calibration Error (ECE).
7. **Interpretabilità** (`grad-cam*.py`, `eigen-cam*.py`, altri script in
   `PythonCode/`) — Grad-CAM/eigen-CAM (libreria `pytorch_grad_cam`) su una classe
   target, con generazione bilanciata di esempi Veri Positivi/Negativi e Falsi
   Positivi/Negativi (10 per categoria) per l'analisi qualitativa in tesi.
8. **Robustezza** (`robustness*.py`) — stress test applicando rumore gaussiano e blur
   alle immagini di test, misura dell'entropia di Shannon delle predizioni come proxy
   dell'incertezza del modello sotto perturbazione.
9. **Analisi errori ed ensemble** (`erroranalysis*.py`) — analisi dei casi mal
   classificati per singolo modello e in configurazione ensemble (es. DenseNet121 +
   ResNet50 con pesi XRV, ciascuno con soglia ottimizzata individualmente); alcuni script
   esplorano anche combinazioni pesate ottimizzate con Optuna o un meta-modello Random
   Forest sopra le predizioni dei singoli modelli.

## Architetture coperte

DenseNet121, DenseNet169, ResNet50, EfficientNet B0/B3/B7, ConvNeXt (base, via `timm`,
pre-addestrato ImageNet-22k, incluse varianti "Debiasing"), SwinV2 (via `transformers`),
DINOv2, Rad-DINO, oltre a diverse combinazioni ensemble delle precedenti.

## Struttura del progetto

Tutti gli script si trovano in `PythonCode/` (struttura originale, non modificata).
Mappa indicativa per categoria funzionale:

| Categoria | Prefisso file | Esempi |
|---|---|---|
| Preparazione/esplorazione dati | vario | `dataset.py`, `datasetXRV.py`, `preprocessing.py`, `cleaning.py`, `split.py`, `DataLeakage.py`, `EsplorazioneCheXpert.py`, `PercentualiPatologie.py`, `CaricamentoPesiBiomedicali.py` |
| Training | `train*.py` | `trainDensenet121.py`, `trainAdvancedDensenet121.py`, `trainConvNeXt-Debiasing.py`, `trainSwinV2-ChestXRay.py`, `trainRad-Dino.py`, ... |
| Valutazione | `eval*.py` | `evalDensenet121.py`, `evalEnsembleConvNeXtSwinV2.py`, ... |
| Ottimizzazione soglie | `threshold*.py` | `thresholdResnet50.py`, `thresholdAdvancedDensenet121(2).py`, ... |
| Analisi errori / ensemble | `erroranalysis*.py` | `erroranalysisDensenet121(2).py`, `erroranalysisEnsembleDensenet121Resnet50-xrv.py`, `erroranalysisEnsembleOptunaDensenet121Resnet50-xrv.py`, ... |
| Calibrazione | `calibration*.py` | `calibrationConvNeXt.py`, `calibration-robustnessConvNeXt-Debiasing.py` |
| Interpretabilità | `grad-cam*.py`, `eigen-cam*.py` | `grad-camConvNeXt.py`, `eigen-camSwinV2.py`, `ActivationMaximationConvNeXt.py`, `DifferenceMapConvNeXt.py`, `GuidedActivationConvNeXt.py` |
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
Alcuni script più recenti usano invece `BASE_DIR = os.path.dirname(os.path.abspath(__file__))`,
che è portabile.

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

## Come riprodurre l'ambiente

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Nota**: `requirements.txt` include alcune dipendenze CUDA/NVIDIA specifiche
> dell'ambiente originale (GPU e driver della VM su cui è stato sviluppato il progetto).
> Su una macchina con GPU/driver diversi, o senza GPU, potrebbe essere necessario
> adattare le versioni di `torch` e dei pacchetti `nvidia-*`.

## Ordine tipico di esecuzione

Dedotto dalla logica della pipeline (non verificato con un run end-to-end):

1. `cleaning.py` — valida il dataset scaricato
2. `split.py` — genera `train/val/test_split.csv` (poi `DataLeakage.py` per verifica)
3. uno script `train*.py` (baseline o "Advanced") per il modello scelto
4. uno script `threshold*.py` corrispondente, sul checkpoint prodotto
5. `eval*.py`, `calibration*.py`, `grad-cam*.py` / `eigen-cam*.py`,
   `robustness*.py`, `erroranalysis*.py` a valle, sullo stesso checkpoint

## Autore

Luca Migliaccio — tesi seguita da Antonio Esposito, Università degli Studi della Campania
"Luigi Vanvitelli".
