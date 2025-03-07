#!/bin/bash


eval "$(mamba shell hook --shell bash)" && mamba activate ai_ta_hpo
source ./finetune/scripts/set/env/set_vars.sh


lr=1e-3
wandb_run_id="" # enable resume by setting this to the run id
timestamp=$(date +"%Y-%m-%d")
run_name="hpo_lr_${lr}_${timestamp}"
output_dir="${OUTPUT_DIR}/optuna/${run_name}"
logging_dir="${LOGGING_DIR}/optuna/${run_name}"

# avoid optuna internal error about setting hpo decision space
if [ "${wandb_run_id}" == "" ]; then
    rm -rf "${output_dir}" && mkdir -p "${output_dir}"
    rm -rf "${logging_dir}" && mkdir -p "${logging_dir}"
fi

model_name="meta-llama/Llama-3.2-11B-Vision-Instruct" # main model
#model_name="meta-llama/Llama-3.2-1B" # for pipeline testing

start_time=$(date +%s)

echo "Run finetune hpo with deepspeed z3 with lr=${lr}"
version_dir="./finetune/hpo"
python "${version_dir}/main.py" \
    --seed 42 \
    --n_trials 54 \
    --lr "${lr}" \
    --model_name_or_path "${model_name}" \
    --no-use_4bit_quantization \
    --data_file "./data/hpo/qa_with_chat_template.csv" \
    --accelerate_config "${version_dir}/config/ds_zero3_expanse.yaml" \
    --run_name "${run_name}" \
    --output_dir "${output_dir}" \
    --logging_dir "${logging_dir}" \
    --wandb_project "ai_ta_finetune" \
    --wandb_name "${run_name}" \
    --wandb_api_key "${WANDB_API_KEY}" \
    --hf_token "${HF_TOKEN}" \
    --wandb_run_id "${wandb_run_id}"

end_time=$(date +%s)
tot_time=$((end_time - start_time))
tot_time=$((tot_time / 60))
echo "Elapsed time: ${tot_time} mins"

echo "Check ${logging_dir} for other logs"
echo "Check ${output_dir} for the output"