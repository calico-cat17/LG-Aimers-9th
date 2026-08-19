#!/usr/bin/env bash
set -euo pipefail
output_name="${1:-submit.zip}"

if [[ ! -f model/deep_manifest.json && ! -f model/model.joblib ]]; then
  echo "trained model is missing; run: python train_deep.py (or python train.py)" >&2
  exit 1
fi

python -m py_compile script.py src/features.py src/deep_model.py src/deep_preprocessing.py
zip -r "$output_name" model src script.py requirements.txt \
  -x '*/__pycache__/*' '*.pyc' 'model/validation_*.npz' \
     'model/model_all_seasons.joblib'
unzip -l "$output_name"
