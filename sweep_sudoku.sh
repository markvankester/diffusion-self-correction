#!/bin/bash
#SBATCH --job-name=sweep_sudoku
#SBATCH --time=05:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --array=0-3        # 0=remdm, 1=prism, 2=backplay, 3=remedi

source $HOME/.bashrc
cd $HOME/diffusion-self-correction
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

CHECKPOINTS=("checkpoints/sudoku/baseline/Run 2/checkpoint-200000" 
             "checkpoints/sudoku/prism/checkpoint-100000"
             "checkpoints/sudoku/backplay/checkpoint-250000"
             "checkpoints/sudoku/remedi/checkpoint-350000")
METHODS=("remdm_conf" "prism" "backplay" "remedi")

uv run python scripts/run_hyperparam_sweep.py \
    --task sudoku \
    --checkpoint "${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}" \
    --methods ${METHODS[$SLURM_ARRAY_TASK_ID]//,/ } \
    --num_prompts 500 \
    --output sweep_results/sudoku_sweep_${SLURM_ARRAY_TASK_ID}.csv \
    --save_examples
