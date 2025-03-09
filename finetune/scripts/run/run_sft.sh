#!/bin/bash


MAMBA_ENV="ai_ta_hpo"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source ./finetune/scripts/set/env/set_vars.sh

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

MODEL_VERSION=TOMMI-0.2-auto

PY_SCRIPT=./finetune/sft/sft_trl.py # the correct version for fine-tuning with instruct model
#PY_SCRIPT=./finetune/sft/sft_navie.py # the inner core of hpo legacy, which should not be used anymore

PY_CONFIG=./finetune/recipes/sft/"${MODEL_VERSION}".yaml

echo ""
echo "Accelerating script: ${PY_SCRIPT} with config: ${PY_CONFIG}"
echo ""

ACCEL_CONFIG=./finetune/recipes/accelerate_ds_cfgs/ds_zero2_carc.yaml
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --num_processes="${GPU_COUNT}" \
    --config_file "${ACCEL_CONFIG}" "${PY_SCRIPT}" --config "${PY_CONFIG}"

echo "END TIME: $(date)"
echo "DONE"
