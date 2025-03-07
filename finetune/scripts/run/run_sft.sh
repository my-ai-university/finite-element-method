#!/bin/bash


MAMBA_ENV="ai_ta_hpo"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source ./finetune/scripts/set/env/set_vars.sh

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

MODEL_VERSION=TOMMI-book-qa-test

PY_SCRIPT=./finetune/sft/sft_trl.py # an advanced and more efficient version
PY_CONFIG=./finetune/recipes/trl/"${MODEL_VERSION}".yaml

# PY_SCRIPT=./finetune/sft/sft_plain.py # the inner core of hpo
# PY_CONFIG=./finetune/recipes/"${MODEL_VERSION}".yaml

ACCEL_CONFIG=./finetune/recipes/accelerate_ds_cfgs/ds_zero3_carc.yaml

echo ""
echo "Accelerating script: ${PY_SCRIPT} with config: ${PY_CONFIG}"
echo ""

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --num_processes="${GPU_COUNT}" \
    --config_file "${ACCEL_CONFIG}" "${PY_SCRIPT}" --config "${PY_CONFIG}"

echo "END TIME: $(date)"
echo "DONE"
