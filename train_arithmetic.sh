#!/bin/bash
#SBATCH --job-name=mdlm_arithmetic
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1

source $HOME/.bashrc
cd $HOME/diffusion-self-correction

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

uv run MDLM/train_mdlm.py --config MDLM/configs/train_mdlm_arithmetic.toml
