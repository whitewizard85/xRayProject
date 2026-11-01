#!/usr/bin/env bash
# =====================================================================
# 01_download_data.sh -- scarica i dataset in locale.
#
#   NIH ChestX-ray14 : se la cartella esiste gia' (Data_Entry_2017.csv presente) viene SALTATO.
#                      Altrimenti via Kaggle API (serve ~/.kaggle/kaggle.json), ~45 GB.
#   VinDr-CXR        : da PhysioNet, accesso CREDENZIALATO (vedi README, passo 0).
#                      Scarica SOLO annotations/ e test/ (3000 DICOM) -- non il train.
#                      Credenziali lette da ~/.netrc (nessuna password nello script).
#
# Uso:   bash 01_download_data.sh            (legge i percorsi da config.yaml)
# Si puo' rilanciare: wget -c / -N riprende i download interrotti.
# =====================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CFG="${XRAY_CONFIG:-$HERE/config.yaml}"

get() { python3 -c "import yaml,sys; c=yaml.safe_load(open('$CFG')); print(c['paths']['$1'])"; }
NIH_DIR="$(get nih_dir)"
VINDR_ROOT="$(get vindr_root)"
VINDR_SOURCE="$(get vindr_source)"
VINDR_KAGGLE="$(get vindr_kaggle_dir)"

echo "== NIH ChestX-ray14 -> $NIH_DIR"
if [[ -f "$NIH_DIR/Data_Entry_2017.csv" && -f "$NIH_DIR/BBox_List_2017.csv" ]]; then
  echo "   gia' presente, salto."
else
  command -v kaggle >/dev/null || { echo "   ERRORE: 'kaggle' CLI non trovato (pip install kaggle) e NIH assente."; exit 1; }
  mkdir -p "$NIH_DIR"
  kaggle datasets download -d nih-chest-xrays/data -p "$NIH_DIR" --unzip
fi
n_nih=$(find "$NIH_DIR" -path '*images_*' -name '*.png' | wc -l)
echo "   immagini NIH trovate: $n_nih (attese 112120)"

if [[ "$VINDR_SOURCE" == "kaggle" ]]; then
  echo "== VinDr-CXR (competizione Kaggle VinBigData) -> $VINDR_KAGGLE"
  echo "   NB: questo scarica lo ZIP COMPLETO (centinaia di GB). Con poco disco usa invece:"
  echo "       python 01b_download_vindr_kaggle_stream.py   (un'immagine alla volta)"
  command -v kaggle >/dev/null || { echo "   ERRORE: 'kaggle' CLI non trovato (pip install kaggle)."; exit 1; }
  [[ -f "$HOME/.kaggle/kaggle.json" || -f "$HOME/.kaggle/access_token" || -n "${KAGGLE_API_TOKEN:-}${KAGGLE_KEY:-}" ]] || {
    echo "   ERRORE: credenziali Kaggle assenti. Kaggle -> Settings -> API -> 'Create New Token',"
    echo "   poi salva il file in ~/.kaggle/kaggle.json (chmod 600)."; exit 1; }
  mkdir -p "$VINDR_KAGGLE"
  C=vinbigdata-chest-xray-abnormalities-detection
  # Serve aver cliccato "I Understand and Accept" nella pagina Rules della competizione (errore 403 altrimenti).
  if [[ ! -f "$VINDR_KAGGLE/train.csv" ]]; then
    kaggle competitions download -c "$C" -p "$VINDR_KAGGLE"
    echo "   estrazione (solo train.csv e train/)..."
    unzip -q -n "$VINDR_KAGGLE/$C.zip" 'train.csv' 'train/*' -d "$VINDR_KAGGLE"
    echo "   puoi cancellare $VINDR_KAGGLE/$C.zip per liberare spazio."
  fi
  n_vd=$(find "$VINDR_KAGGLE/train" -iname '*.dicom' | wc -l)
  echo "   DICOM VinDr (Kaggle train) trovati: $n_vd (attesi 15000)"
  echo "== Download completato. Prossimo passo: python 02_prepare_data.py"
  exit 0
fi

echo "== VinDr-CXR (PhysioNet) -> $VINDR_ROOT/vindr-cxr/1.0.0"
if ! grep -q "physionet.org" "$HOME/.netrc" 2>/dev/null; then
  cat <<EOF
   ERRORE: credenziali PhysioNet non trovate in ~/.netrc.
   Crea il file cosi' (una volta sola):
       echo "machine physionet.org login TUO_USERNAME password TUA_PASSWORD" >> ~/.netrc
       chmod 600 ~/.netrc
   Prima devi: account PhysioNet credenzialato + training CITI + firma DUA del progetto vindr-cxr.
EOF
  exit 1
fi
mkdir -p "$VINDR_ROOT"
BASE="https://physionet.org/files/vindr-cxr/1.0.0"

fetch_dir() {  # prova il nome minuscolo e, se fallisce, quello con l'iniziale maiuscola
  local d="$1"
  for name in "$d" "${d^}"; do
    if wget -r -N -c -np -nH --cut-dirs=1 -e robots=off --reject "index.html*" \
         -P "$VINDR_ROOT" "$BASE/$name/"; then
      return 0
    fi
  done
  echo "   ERRORE: impossibile scaricare la cartella '$d'. Controlla accesso/DUA su PhysioNet."
  return 1
}
fetch_dir annotations
fetch_dir test

n_vd=$(find "$VINDR_ROOT/vindr-cxr/1.0.0" -iname '*.dicom' -o -iname '*.dcm' | wc -l)
echo "   DICOM VinDr trovati: $n_vd (attesi 3000 per il test set)"
echo "== Download completato. Prossimo passo: python 02_prepare_data.py"
