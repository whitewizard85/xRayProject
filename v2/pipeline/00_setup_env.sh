#!/usr/bin/env bash
# 00_setup_env.sh -- aggiunge al venv ESISTENTE del progetto i pacchetti mancanti.
# torch/timm/sklearn/pandas sono gia' nel venv (requirements.txt del repo); qui solo gli extra.
# Uso:  bash 00_setup_env.sh /percorso/al/venv     (default: "/home/gpuvm/Desktop/Luca Migliaccio/xrv_env")
set -euo pipefail
VENV="${1:-/home/gpuvm/Desktop/Luca Migliaccio/xrv_env}"
HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "venv non trovato in $VENV: lo creo nuovo (dovrai installare anche torch per CUDA 13)."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -r "$HERE/requirements_full.txt"
fi
"$VENV/bin/pip" install -r "$HERE/requirements_extra.txt"
"$VENV/bin/python" - <<'PY'
import torch, timm, pydicom, statsmodels, yaml
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("timm", timm.__version__, "| pydicom", pydicom.__version__, "| statsmodels", statsmodels.__version__)
PY
echo "OK. Attiva con: source \"$VENV/bin/activate\""
