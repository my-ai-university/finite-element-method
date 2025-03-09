#!/bin/bash


MAMBA_ENV="ai_ta_hpo"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source ./finetune/scripts/set/env/set_vars.sh

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

MODEL_VERSION=TOMMI-hpo # TOMMI-elephant-hpo
DATA_FILE=./data/hpo/qa_250201.csv # ./data/hpo/qa_elephant.csv

PY_SCRIPT=./finetune/hpo/main.py
PY_LAUNCH_SCRIPT=./finetune/hpo/train.py
ACCEL_CONFIG=./finetune/recipes/accelerate_ds_cfgs/ds_zero2_carc.yaml

export ACCELERATE_LOG_LEVEL=info

echo ""
echo "Accelerating TOMMI hpo with data: ${DATA_FILE}"
echo ""

python "${PY_SCRIPT}" \
    --n_trials 54 \
    --num_processes="${GPU_COUNT}" \
    --run_name="${MODEL_VERSION}" \
    --data_file="${DATA_FILE}" \
    --accelerate_config "${ACCEL_CONFIG}" \
    --py_launch_script "${PY_LAUNCH_SCRIPT}" \
    --wandb_project "finite-element-method"

echo "END TIME: $(date)"
echo "DONE"
