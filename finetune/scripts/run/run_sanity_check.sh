#!/bin/bash


eval "$(mamba shell hook --shell bash)" && mamba activate ai_ta_hpo
source ./finetune/scripts/set/env/set_vars.sh


python ./finetune/evaluate/sanity_check.py