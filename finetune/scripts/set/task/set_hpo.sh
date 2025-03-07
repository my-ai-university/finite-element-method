#!/bin/bash
# python 3.10 + cuda 11.8.0


export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

conda clean -a -y # conda for traditional and reliable setup
mamba clean -a -y # mamba for smart and efficient setup
pip install --upgrade pip

# cuda, gcc/g++, torch
conda install cuda -c nvidia/label/cuda-11.8.0 -y
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install torchao==0.7.0 --index-url https://download.pytorch.org/whl/cu118

# deepspeed
mamba install gcc gxx -c conda-forge -y # ensure > 9.0 for ninja JIT
pip install deepspeed==0.15.4

# bitsandbytes
pip install setuptools
mamba install bitsandbytes=0.45.0 -c conda-forge --no-deps -y
pip install psutil
# add the following to your .bashrc or running scripts
#export BNB_CUDA_VERSION=118
#export CUDA_HOME=$CONDA_PREFIX
#export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# trl, accelerate, peft
pip install trl
pip install accelerate peft optuna datasets

# other dependencies
pip install scikit-learn pexpect
pip install wandb plotly # takes a while