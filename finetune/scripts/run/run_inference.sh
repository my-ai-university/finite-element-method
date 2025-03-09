#!/bin/bash


MAMBA_ENV="ai_ta_hpo"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source ./finetune/scripts/set/env/set_vars.sh

python ./finetune/evaluate/plain_inference.py

echo "END TIME: $(date)"
echo "DONE"
