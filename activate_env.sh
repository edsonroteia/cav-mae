#!/bin/bash
# CAV-MAE environment activation script

# Activate the virtual environment
source /weka/kuehne/kqr867/code/cav-mae/.venv/bin/activate

# Set CUDA paths for H100 compute nodes
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/2023/cuda/12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Set torch home for pretrained models
export TORCH_HOME=/weka/kuehne/kqr867/code/cav-mae/pretrained_models
