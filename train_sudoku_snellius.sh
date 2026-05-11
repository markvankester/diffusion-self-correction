#!/bin/bash
#SBATCH --job-name=mdlm_sudoku
#SBATCH --time=00:10:00           
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1              

# Ensure your PATH is loaded just in case `uv` is installed in ~/.local/bin
source $HOME/.bashrc

# Navigate to your project directory
cd $HOME/diffusion-self-correction 

# Ensure the directory where `uv` is installed is in the PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# uv run automatically resolves the virtual environment and runs the script
uv run MDLM/train_mdlm.py MDLM/configs/train_sudoku.toml
