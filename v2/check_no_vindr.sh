#!/usr/bin/env bash
# Run from the repository root before every push. Fails if anything that looks like VinDr-CXR data is staged.
set -e
bad=$(git diff --cached --name-only | grep -Ei '\.(dicom|dcm|npz)$|(^|/)vindr_[^/]*\.csv$|(^|/)train\.csv$|(^|/)cache/|preds_' || true)
big=$(git diff --cached --name-only | while read -r f; do [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 5000000 ] && echo "$f"; done || true)
if [ -n "$bad$big" ]; then
  echo "STOP: these staged files must not be pushed:"; echo "$bad"; echo "$big"; exit 1
fi
echo "OK: no VinDr data or large files staged."
