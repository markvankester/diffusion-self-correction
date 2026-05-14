"""
train_mdlm.py
=============
Standalone training script for Masked Diffusion Language Models (MDLM).

All hyperparameters are controlled via task-specific TOML config files.
CLI flags override any TOML value if provided.
"""

from pathlib import Path
import sys
import os
import tomllib

# Allow running directly as a script from any working directory
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq


from backbones.llada.config import MDLMConfig
from backbones.llada.model import MDLMModelLM
from diffusion.schedules import LinearAlphaScheduler
from diffusion.trainer import MDLMConfig as TrainerConfig, MDLMTrainer
from diffusion.sampler import MDLMSampler, MDLMSamplerConfig
from data.processing.collators import NoAttentionMaskWrapper
from MDLM.tasks import get_task_adapter


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "train_arithmetic.toml"
TASK_CONFIG_PATHS = {
    "arithmetic": CONFIG_DIR / "train_arithmetic.toml",
    "sudoku": CONFIG_DIR / "train_sudoku.toml",
}


# ============================================================================
# Config / Argument Parsing
# ============================================================================

def load_config_file(config_path: str | os.PathLike) -> dict:
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level TOML table.")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Masked Diffusion Language Model")

    parser.add_argument("--config", type=str, default=None,
                        help="Path to a TOML config file")
    parser.add_argument("--task", type=str, choices=["arithmetic", "sudoku"], default=None,
                        help="Task adapter used for dataset and tokenizer defaults.")

    # Data & Paths
    parser.add_argument("--data_path",    type=str,   default=None)
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--output_dir",   type=str,   default=None)
    parser.add_argument("--tokenizer_id", type=str,   default=None)
    parser.add_argument("--limit_data",   type=int,   default=None,
                        help="Limit number of rows loaded (0 for all)")
    parser.add_argument("--mask_until_token", type=str, default=None,
                        help="Token string up to which inputs will be masked during training (e.g. '=').")

    # Model Architecture
    parser.add_argument("--d_model",   type=int, default=None)
    parser.add_argument("--n_heads",   type=int, default=None)
    parser.add_argument("--n_layers",  type=int, default=None)
    parser.add_argument("--mlp_ratio", type=int, default=None)
    parser.add_argument("--seq_len",   type=int, default=None)

    # Advanced Architecture
    parser.add_argument("--layer_norm_type", type=str, default=None)
    parser.add_argument("--activation_type", type=str, default=None)
    parser.add_argument("--block_type",      type=str, default=None)
    parser.add_argument("--init_fn",         type=str, default=None)
    parser.add_argument("--attention_dropout", type=float, default=None)
    parser.add_argument("--residual_dropout", type=float, default=None)
    parser.add_argument("--embedding_dropout", type=float, default=None)
    parser.add_argument("--weight_tying", action=argparse.BooleanOptionalAction, default=None)

    # Training Hyperparameters
    parser.add_argument("--batch_size",   type=int,   default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--grad_accum",   type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--lr_scheduler_type", type=str, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_steps",    type=int,   default=None)
    parser.add_argument("--warmup_steps", type=int,   default=None)
    parser.add_argument("--save_steps",   type=int,   default=None)
    parser.add_argument("--eval_strategy", type=str, default="no")
    parser.add_argument("--eval_steps",   type=int,   default=None)
    parser.add_argument("--logging_steps", type=int,  default=None)
    parser.add_argument("--run_name",     type=str,   default=None)
    parser.add_argument("--time_epsilon", type=float, default=None)
    parser.add_argument("--loss_norm_type", type=str, default=None)
    # Misc
    parser.add_argument("--report_to",   type=str, default=None,
                        help="Experiment tracking backend ('wandb' or 'none')")
    parser.add_argument("--loss_weight_type", type=str, default=None,
                        help="Loss weighting scheme ('scheduler' or 'uniform')")
    parser.add_argument("--skip_sample", action=argparse.BooleanOptionalAction, default=None,
                        help="Skip the post-training sampling test")
    parser.add_argument("--eval_fraction", type=float, default=None,
                        help="Fraction of training data used as eval split (Sudoku only; e.g. 0.05).")

    # Load TOML first, then let CLI flags override
    initial_args, _ = parser.parse_known_args()
    config_path = _default_config_path(initial_args)
    parser.set_defaults(**load_config_file(config_path))
    parser.set_defaults(config=str(config_path))
    return parser.parse_args()


def _default_config_path(args: argparse.Namespace) -> Path:
    if args.config:
        return Path(args.config)
    if args.task in TASK_CONFIG_PATHS:
        return TASK_CONFIG_PATHS[args.task]
    return DEFAULT_CONFIG_PATH


# ============================================================================
# Setup Helpers
# ============================================================================

def build_tokenizer(tokenizer_id: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def build_model(args, tokenizer) -> MDLMModelLM:
    arch_kwargs = {
        k: getattr(args, k)
        for k in (
            "layer_norm_type",
            "activation_type",
            "block_type",
            "init_fn",
            "attention_dropout",
            "residual_dropout",
            "embedding_dropout",
            "weight_tying",
        )
        if getattr(args, k) is not None
    }

    config = MDLMConfig(
        vocab_size=len(tokenizer),
        embedding_size=len(tokenizer),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        max_sequence_length=args.seq_len,
        rope=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        use_cache=False,
        init_device="cpu",
        **arch_kwargs,
    )
    model = MDLMModelLM(config, init_params=True)
    model.resize_token_embeddings(len(tokenizer))
    return model

# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    task = get_task_adapter(args.task or "arithmetic")
    args.task = task.name

    if args.data_path is None:
        args.data_path = task.default_data_path
    if args.tokenizer_id is None:
        args.tokenizer_id = task.default_tokenizer_path

    print(f"\n{'='*60}")
    print(f"  MDLM TRAINING SCRIPT")
    print(f"  Config: {args.config}")
    print(f"  Task:   {args.task}")
    print(f"{'='*60}")

    # 1. Tokenizer
    print(f"\n[1/5] Loading tokenizer ({args.tokenizer_id})...")
    tokenizer = build_tokenizer(args.tokenizer_id)
    print(f"  Vocab size : {len(tokenizer)}")
    print(f"  Mask token : {tokenizer.mask_token_id}")

    # 2. Dataset — delegated to the task adapter for full modularity.
    print(f"\n[2/5] Building dataset via {task.name} adapter ({args.data_path})...")
    dataset, eval_dataset = task.build_datasets(
        tokenizer=tokenizer,
        data_path=args.data_path,
        seq_len=args.seq_len,
        eval_data_path=getattr(args, "eval_data_path", None),
        limit_data=getattr(args, "limit_data", 0) or 0,
        mask_until_token=getattr(args, "mask_until_token", None),
        eval_fraction=getattr(args, "eval_fraction", None) or 0.0,
    )

    # 3. Model
    model = build_model(args, tokenizer)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[3/5] Model ready — {n_params:,} parameters")

    # 4. Trainer
    trainer_config = TrainerConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size if args.eval_batch_size is not None else 1024,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay if args.weight_decay is not None else 0.01,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        run_name=args.run_name,
        report_to=args.report_to,
        time_epsilon=args.time_epsilon,
        loss_weight_type=args.loss_weight_type,
        loss_norm_type=args.loss_norm_type,
        remove_unused_columns=False,
        bf16=True,
        dataloader_num_workers=4,
    )

    scheduler = LinearAlphaScheduler()
    trainer = MDLMTrainer(
        args=trainer_config,
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        scheduler=scheduler,
        data_collator=NoAttentionMaskWrapper(
            DataCollatorForSeq2Seq(
                tokenizer,
                return_tensors="pt",
                padding=True,
                label_pad_token_id=tokenizer.pad_token_id,
            )
        ),
    )
    print(f"\n[4/5] Trainer ready")

    # 5. Train
    print(f"\n{'='*60}")
    print(f"  STARTING TRAINING")
    print(f"{'='*60}\n")
    trainer.train()
    trainer.save_model()

    print(f"\n{'='*60}")
    print(f"  DONE — checkpoints saved to '{args.output_dir}'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
