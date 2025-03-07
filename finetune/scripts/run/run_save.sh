#!/bin/bash


eval "$(mamba shell hook --shell bash)" && mamba activate ai_ta_hpo
source ./finetune/scripts/set/env/set_vars.sh

echo "save the fine-tuned model produced by the best hyperparameters"
python "./finetune/utils/save_finetuned.py"