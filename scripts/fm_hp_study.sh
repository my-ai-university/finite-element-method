#!/bin/bash

#SBATCH --account=garikipa_1359
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          ## was 32
#SBATCH --gpus-per-task=1
#SBATCH --constraint=[a100-40gb|a100-80gb|epyc-7282|epyc-7313]  ## a40 or a100
#SBATCH --time=04:00:00             ## hh:mm:ss
#SBATCH --array=1                   # specify <1-X>
#SBATCH --export=ALL
#SBATCH --output=/project/garikipa_1359/projects/ai_ta/hyperparam_opt/slurm_out/%A_%a.out

module purge

eval "$(conda shell.bash hook)"
conda activate /project/garikipa_1359/envs/llm_env_12

STUDY_ID="hp_opt_241002"
export STUDY_ID

echo "Study ID: $STUDY_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

python3 test_runner.py