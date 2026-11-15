# v2 — Replicated factorial study with external validation

Code and aggregated results for the manuscript *"How reliable are attribution and calibration
conclusions for chest radiograph classifiers? A replicated factorial study with external validation"*
(L. Migliaccio, A. Esposito). This folder is self-contained; the rest of the repository holds the
earlier thesis code and is not needed to reproduce v2.

## Design in brief
- **Training / internal test:** NIH ChestX-ray14, patient-level split reused from
  `../paper_reproducibility/results/patient_splits/` (train/val/test_split.csv).
- **External test only:** VinDr-CXR (Kaggle VinBigData release, 15,000 images, 3 readers, majority label).
- **Factorial block:** crop × loss (BCE/ASL) × schedule (cosine 10 ep / constant LR + early stopping)
  × augmentation (flip/none), 5 seeds, ConvNeXt-Base → 80 runs.
  **Losses block:** weighted BCE, focal (5 seeds each). **Arch block:** DenseNet-121 with BCE/ASL. Total: 100 runs.
- **Outputs per run:** discrimination, calibration (raw / temperature / Platt fitted on NIH val),
  Grad-CAM, Grad-CAM++, Integrated Gradients, localization vs radiologist boxes, deletion test,
  partial randomization test, inter-model map agreement.

## Layout
| Path | Content |
|---|---|
| `pipeline/` | Full pipeline: `01_*` download, `02` preprocessing, `03` training, `04` evaluation, `05` attribution, `06` analysis, `07` numbers quoted in the paper; `run_queue.py` runs everything resumably; `config.yaml` holds all settings. |
| `analysis/make_figures.py` | Figures 1–4 and the schedule-collapsed effect estimates used in the paper (`python make_figures.py results/analysis figures`). |
| `results/analysis/` | Per-run table (`runs_table.csv`, 100 rows), factorial effects and variance decomposition, hierarchical-bootstrap localization effects, agreement, faithfulness/sanity, calibration by loss, collapsed-schedule effects. |
| `results/paper_numbers/` | Selected epoch per run, per-class AUC (NIH vs VinDr) with bootstrap CIs, training time. |
| `figures/` | Figures as in the manuscript. |

All files in `results/` are aggregates over runs or classes; no image-level VinDr data is included.

## Reproducing
```bash
cd pipeline
bash 00_setup_env.sh /path/to/venv          # extra packages on top of torch/timm/sklearn/pandas
bash 01_download_data.sh                     # NIH ChestX-ray14
python 01b_download_vindr_kaggle_stream.py   # VinDr-CXR from Kaggle (account + accepted rules required)
# edit config.yaml: dataset paths, work_dir, splits_dir
python 02_prepare_data.py --config config.yaml   # 512x512 caches, VinDr labels and boxes
python run_queue.py --config config.yaml        # train → evaluate → attribute, resumable
python 06_analyze.py --config config.yaml       # tables in <work_dir>/analysis
python 07_paper_numbers.py config.yaml
python ../analysis/make_figures.py <work_dir>/analysis ../figures
```
Hardware used: one NVIDIA RTX 5000 Ada; 171 GPU-hours of training for the 100 runs.

## Data licences
- NIH ChestX-ray14: public (NIH Clinical Center).
- VinDr-CXR: obtained from the original sources (PhysioNet or the Kaggle VinBigData competition) under their
  non-commercial research terms. **It is not redistributed here.** `check_no_vindr.sh` blocks accidental commits.

## Extension (November 2026) — see `PREREGISTRATION.md`
New blocks and analyses, all resumable through the same queue:

| What | Where | Cost (RTX 5000 Ada, estimate) |
|---|---|---|
| Swin-B at the reference cell, BCE/ASL × 5 seeds (block `swin`) | `experiments.py`, `common.py` | ~25 GPU-h training |
| ConvNeXt-Base, alternative regime (lr 1e-5, wd 0.05, warm-up + cosine 15 epochs, strong augmentation), BCE/ASL × 5 seeds (block `regime`) | `experiments.py`, `03_train.py` | ~30 GPU-h training |
| Full cascading randomization + IG with blurred baseline on the 50 reference-cell models (stage `attrx`) | `05b_attribution_ext.py` | ~15–25 min per model |
| Demographics (NIH partitions; VinDr from DICOM headers streamed from the Kaggle zip) | `09_demographics.py` | no GPU; ~4 GB of traffic for VinDr |
| Analysis of the extension (pre-specified H1–H5 + descriptive tables) | `08_extension_analysis.py` | seconds |

```bash
cd pipeline
python 09_demographics.py --skip-vindr      # NIH only, immediate
python 09_demographics.py                   # also VinDr (needs the Kaggle token, resumable)
python run_queue.py --stages attrx          # extended attribution on the 30 existing reference models (no training)
python run_queue.py                         # trains/evaluates/attributes the 20 new runs, then attrx on them
python 08_extension_analysis.py             # -> <work_dir>/analysis_ext/EXT_REPORT.md
```
