#!/bin/bash


export BNB_CUDA_VERSION=118
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export OUTPUT_PREFIX="/expanse/lustre/projects/mia346/$USER/projects/ai_ta"
export SCRATCH_PREFIX="/expanse/lustre/scratch/$USER/temp_project/projects/ai_ta"
export OUTPUT_DIR="${OUTPUT_PREFIX}/output"
export LOGGING_DIR="${OUTPUT_PREFIX}/logging"

export WANDB_API_KEY=""
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}/wandb"

export HF_TOKEN=""
export HF_HOME="${OUTPUT_PREFIX}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${OUTPUT_PREFIX}/.cache/huggingface/hub"
export HG_DATASETS_CACHE="${OUTPUT_PREFIX}/.cache/huggingface/datasets"

export DS_LOG_LEVEL=error
export TRITON_CACHE_DIR="${OUTPUT_PREFIX}/.cache/triton_cache"
export MKL_THREADING_LAYER=GNU
export TOKENIZERS_PARALLELISM=false

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1

export PYTHONPATH="/home/$USER/workspace/AI-TA/comp-phys-transformer":$PYTHONPATH
export PYTHONPATH="/home/$USER/workspace/AI-TA/comp-phys-transformer/finetune":$PYTHONPATH