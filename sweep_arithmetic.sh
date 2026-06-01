#!/bin/bash
#SBATCH --job-name=sweep_arithmetic
#SBATCH --time=05:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --array=0-1        # 0=baseline+remdm, 1=prism

source $HOME/.bashrc
cd $HOME/diffusion-self-correction
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

CHECKPOINTS=("checkpoints/arithmetic/final2/checkpoint-250000" 
             "checkpoints/arithmetic/prism/checkpoint-100000")
METHODS=("remdm_conf" "prism")

uv run python scripts/run_hyperparam_sweep.py \
    --task arithmetic \
    --checkpoint "${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}" \
    --methods ${METHODS[$SLURM_ARRAY_TASK_ID]//,/ } \
    --num_prompts 1000 \
    --dataset_path data/arithmetic_test_corrupted.jsonl \
    --output sweep_results/arithmetic_sweep_corrupted_${SLURM_ARRAY_TASK_ID}.csv \
    --save_examples


