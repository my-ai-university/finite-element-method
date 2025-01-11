#!/bin/bash

#SBATCH --job-name=ai_ta_gpo_test
#SBATCH --output=/expanse/lustre/projects/mia346/swang31/projects/ai_ta/output/slurm/%A_%a.out
#SBATCH --partition=gpu
#SBATCH --account=mia346
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00

./finetune/scripts/run/run_hpo.sh
