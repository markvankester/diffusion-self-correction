#!/bin/bash
#SBATCH --job-name=mdlm_sudoku
#SBATCH --time=00:10:00           
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_shared           
#SBATCH --gres=gpu:1              

# Ensure your PATH is loaded just in case `uv` is installed in ~/.local/bin
source $HOME/.bashrc

# Navigate to your project directory
cd $HOME/diffusion-self-correction 

# uv run automatically resolves the virtual environment and runs the script
uv run MDLM/train_mdlm.py MDLM/configs/train_sudoku.toml
