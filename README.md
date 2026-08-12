# Chest X-Ray Classification — Tesi Migliaccio

> **Nota**: questa descrizione è stata ricostruita a partire dai nomi dei file e dalla struttura
> delle cartelle, non dal contenuto del codice. Verifica e correggi le parti che non
> corrispondono al lavoro reale prima di considerarlo definitivo.

## Descrizione

Progetto di classificazione automatica di radiografie del torace (chest X-ray) tramite
modelli di deep learning, sviluppato nell'ambito della tesi di Luca Migliaccio.
Il lavoro sembra includere addestramento e confronto di più architetture, tecniche di
interpretabilità (Grad-CAM), calibrazione delle probabilità, analisi di robustezza e
analisi degli errori.

**Dataset utilizzati (non inclusi nel repository — vedi sotto):**
- CheXpert
- NIH ChestX-ray14 ("ChestXray14" / `archive/`)

**Architetture coinvolte (dedotte dai nomi dei file):**
- DenseNet121, DenseNet169
- ResNet50
- EfficientNet B0 / B3 / B7
- ConvNeXt
- SwinV2
- DINOv2, Rad-DINO
- Modelli ensemble (combinazioni delle precedenti)
- Alcuni modelli sembrano pre-addestrati su dataset biomedicali (XRV / weights biomedicali)

## Struttura del progetto

Tutti gli script si trovano in `PythonCode/` (struttura originale, non modificata).
Ecco una mappa indicativa per orientarsi, per categoria funzionale:

| Categoria | Prefisso file | Esempi |
|---|---|---|
| Preparazione/esplorazione dati | vario | `dataset.py`, `datasetXRV.py`, `preprocessing.py`, `cleaning.py`, `split.py`, `DataLeakage.py`, `EsplorazioneCheXpert.py`, `PercentualiPatologie.py`, `CaricamentoPesiBiomedicali.py` |
| Training | `train*.py` | `trainDensenet121.py`, `trainConvNeXt.py`, `trainSwinV2-ChestXRay.py`, ... |
| Valutazione | `eval*.py` | `evalDensenet121.py`, `evalEnsembleConvNeXtSwinV2.py`, ... |
| Ottimizzazione soglie | `threshold*.py` | `thresholdResnet50.py`, `thresholdAdvancedDensenet121(2).py`, ... |
| Analisi errori | `erroranalysis*.py` | `erroranalysisDensenet121(2).py`, `erroranalysisEnsembleDensenet121Resnet50-xrv.py`, ... |
| Calibrazione | `calibration*.py` | `calibrationConvNeXt.py`, `calibration-robustnessConvNeXt-Debiasing.py` |
| Interpretabilità | `grad-cam*.py`, `eigen-cam*.py` | `grad-camConvNeXt.py`, `eigen-camSwinV2.py`, `ActivationMaximationConvNeXt.py`, `DifferenceMapConvNeXt.py`, `GuidedActivationConvNeXt.py` |
| Robustezza | `robustness*.py` | `robustnessConvNeXt.py`, `robustnessConvNeXtDebiasing2.py` |
| Grafici riassuntivi | vario | `GenerazioneGraficoRobustezza.py`, `GenerazioneIstogrammi.py` |
| Script esplorativi/prove | `prova*.py` | `prova.py`, `prova1.py`, `provaEnsemble.py`, `analysisSwinV2.py` |

## Dati e file esclusi dal repository

Questo repository contiene **solo il codice**. Sono esclusi (vedi `.gitignore`):
- I dataset (`archive/`, `archiveCheXpert/`) — diverse decine di GB
- Il virtual environment Python (`xrv_env/`)
- Checkpoint / pesi dei modelli addestrati
- Output di analisi generati (grafici, immagini Grad-CAM, report)

> **Da verificare**: non è ancora presente un `requirements.txt` / `environment.yml` con le
> dipendenze esatte. Consiglio di generarlo dalla VM con l'ambiente attivo:
> ```bash
> source xrv_env/bin/activate
> pip freeze > requirements.txt
> ```
> così chiunque cloni il repo può ricreare l'ambiente senza il venv originale.

## Come riprodurre l'ambiente

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # da generare, vedi sopra
```

## Come usare gli script

> **Da completare**: aggiungere qui una breve descrizione di come lanciare training,
> valutazione e analisi (parametri principali, path dei dati attesi, output previsti),
> perché al momento non è possibile dedurlo con certezza dai soli nomi dei file.

## Autore

Luca Migliaccio — tesi seguita da Antonio Esposito, Università degli Studi della Campania
"Luigi Vanvitelli".
