#!/bin/bash


export CUDA_LAUNCH_BLOCKING=1
export DS_LOG_LEVEL=error
export TOKENIZERS_PARALLELISM=false

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1

export MKL_THREADING_LAYER=GNU
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

export BNB_CUDA_VERSION=118
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"


# the USC CARC Discovery env
export CLUSTER_NAME="usc discovery"
export PROJECT_ACCOUNT="neiswang_1391" # TODO
export HOME_PREFIX="/home1/${USER}/workspace"
export PROJECT_PREFIX="/project/${PROJECT_ACCOUNT}/${USER}"
export SCRATCH_PREFIX="/scratch1/${USER}"

# # the UCSD ACCESS Expanse env
# export CLUSTER_NAME="access expanse"
# export PROJECT_ACCOUNT="wis189" # TODO
# export HOME_PREFIX="/home/${USER}/workspace"
# export PROJECT_PREFIX="/expanse/lustre/projects/${PROJECT_ACCOUNT}/${USER}"
# export SCRATCH_PREFIX="/expanse/lustre/scratch/${USER}"

export PROJECT_NAME="ai-ta"
export TOPIC_NAME="comp-phys-transformer"
export CORE_POSTFIX="finetune"
export PROJECT_POSTFIX="${PROJECT_NAME}/${TOPIC_NAME}"
export PROJECT_DIR="${PROJECT_PREFIX}/${PROJECT_POSTFIX}"
export PYTHONPATH="${HOME_PREFIX}/${PROJECT_POSTFIX}":$PYTHONPATH
export PYTHONPATH="${HOME_PREFIX}/${PROJECT_POSTFIX}/${CORE_POSTFIX}":$PYTHONPATH
mkdir -p "${HOME_PREFIX}/${PROJECT_NAME}"

export CKPT_DIR="${PROJECT_DIR}/ckpts"
export DATA_DIR="${PROJECT_DIR}/datasets"
export OUTPUT_DIR="${PROJECT_DIR}/outputs"
export LOGGING_DIR="${PROJECT_DIR}/logs"
mkdir -p "${CKPT_DIR}" "${DATA_DIR}" "${OUTPUT_DIR}" "${LOGGING_DIR}"

export WANDB_API_KEY="" # TODO
export WANDB_PROJECT="${TOPIC_NAME}"
export WANDB_DIR="${OUTPUT_DIR}"
wandb login $WANDB_API_KEY

export CACHE_DIR="${PROJECT_DIR}/.cache"
export TRITON_CACHE_DIR="${CACHE_DIR}/triton_cache"

export HF_TOKEN="" # TODO
huggingface-cli login --token $HF_TOKEN --add-to-git-credential

export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
