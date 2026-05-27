#!/bin/bash
#SBATCH --job-name=backplay_arithmetic
#SBATCH --time=08:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1

source $HOME/.bashrc
cd $HOME/diffusion-self-correction

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

uv run MDLM/train_backplay.py --config MDLM/configs/backplay_arithmetic.toml
