"""
train_prism.py
==============
Fine-tune a pretrained MDLM with the PRISM quality head.
"""

from pathlib import Path
import sys
import os
import argparse

import torch
import torch.nn as nn
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast

# Allow running directly as a script from any working directory
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from MDLM.utils import load_config_file
from backbones.llada.config import MDLMConfig
from backbones.llada.model import MDLMModelLM
from diffusion.schedules import LinearAlphaScheduler
from diffusion.prism import PRISMHead, PRISMTrainer, PRISMConfig
from data.processing.collators import NoAttentionMaskWrapper
from MDLM.tasks import get_task_adapter

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "train_prism_arithmetic.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MDLM with PRISM")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--task", type=str, choices=["arithmetic", "sudoku"], default=None,
                        help="Task adapter used for dataset defaults.")
    
    # Data & Paths
    parser.add_argument("--data_path",    type=str,   default=None)
    parser.add_argument("--output_dir",   type=str,   default=None)
    parser.add_argument("--limit_data",   type=int,   default=None)
    parser.add_argument("--checkpoint",   type=str,   default=None)
    parser.add_argument("--mask_until_token", type=str, default=None)
    parser.add_argument("--eval_data_path", type=str, default=None)

    # PRISM Hyperparameters
    parser.add_argument("--prism_lambda", type=float, default=5.0)
    parser.add_argument("--prism_k",      type=int,   default=4)
    parser.add_argument("--prism_head_type", type=str, default="linear")
    parser.add_argument("--prism_head_n_heads", type=int, default=4)
    parser.add_argument("--prism_head_dropout", type=float, default=0.0)
    parser.add_argument(
        "--prism_freeze_unmasking_head",
        type=lambda v: str(v).lower() in {"1", "true", "yes", "on"},
        default=True,
    )

    # Training & Eval
    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--grad_accum",   type=int,   default=1)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_steps",    type=int,   default=50000)
    parser.add_argument("--warmup_steps", type=int,   default=1000)
    parser.add_argument("--save_steps",   type=int,   default=50000)
    parser.add_argument("--logging_steps", type=int,  default=10)
    parser.add_argument("--eval_strategy", type=str,  default="steps")
    parser.add_argument("--eval_steps",   type=int,   default=1000)
    parser.add_argument("--eval_fraction", type=float, default=0.0,
                        help="Fraction of training data used as eval split when supported by the task adapter.")
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--run_name",     type=str,   default=None)
    parser.add_argument("--time_epsilon", type=float, default=1e-3)
    parser.add_argument("--loss_norm_type", type=str, default="token")
    parser.add_argument("--loss_weight_type", type=str, default="scheduler")
    parser.add_argument("--report_to",    type=str,   default="wandb")
    parser.add_argument("--seq_len",      type=int,   default=None)

    # Load TOML first, then let CLI flags override
    initial_args, _ = parser.parse_known_args()
    parser.set_defaults(**load_config_file(initial_args.config))
    parser.set_defaults(config=initial_args.config)
    return parser.parse_args()

def main():
    args = parse_args()
    task = get_task_adapter(args.task or "arithmetic")
    args.task = task.name

    if args.data_path is None:
        args.data_path = task.default_data_path

    checkpoint_path = args.checkpoint

    print(f"\n[1/4] Loading pretrained MDM for task={args.task} from {checkpoint_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint_path)
    
    # Load model from checkpoint
    checkpoint_config = MDLMConfig.from_pretrained(checkpoint_path)
    print(
        "  Loaded checkpoint config: "
        f"block_type={checkpoint_config.block_type}, "
        f"d_model={checkpoint_config.d_model}, "
        f"n_layers={checkpoint_config.n_layers}, "
        f"n_heads={checkpoint_config.n_heads}"
    )
    model = MDLMModelLM.from_pretrained(checkpoint_path, config=checkpoint_config)
    model.train()
    seq_len = args.seq_len if args.seq_len is not None else checkpoint_config.max_sequence_length

    # Sync diffusion training args from checkpoint to ensure consistency
    training_args_path = Path(checkpoint_path) / "training_args.bin"
    if training_args_path.exists():
        print("  [*] Automatically syncing diffusion loss constraints from checkpoint...")
        try:
            pretrained_args = torch.load(training_args_path, map_location="cpu", weights_only=False)
            if hasattr(pretrained_args, "time_epsilon"):
                args.time_epsilon = pretrained_args.time_epsilon
            if hasattr(pretrained_args, "loss_weight_type"):
                args.loss_weight_type = pretrained_args.loss_weight_type
            if hasattr(pretrained_args, "loss_norm_type"):
                args.loss_norm_type = pretrained_args.loss_norm_type
        except Exception as e:
            print(f"  [!] Failed to load training_args.bin for syncing: {e}")

    # Step 1: Attach PRISM Head
    print(f"[2/4] Attaching PRISM Head...")
    prism_head = PRISMHead(
        d_model=model.config.d_model,
        head_type=args.prism_head_type,
        n_heads=args.prism_head_n_heads,
        dropout=args.prism_head_dropout,
    )
    
    # Move head to same device as model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prism_head.to(device)

    if args.prism_freeze_unmasking_head:
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None:
            for param in output_embeddings.parameters():
                param.requires_grad = False
        elif getattr(model.config, "weight_tying", False):
            print(
                "  WARNING: output weights are tied to input embeddings; "
                "cannot freeze only the unmasking head without changing the backbone architecture."
            )

    prism_head_params = prism_head.num_parameters()
    print(f"  PRISM head parameters: {prism_head_params:,}")

    # Step 2: Configure Trainer
    print(f"[3/4] Setting up PRISM Trainer...")
    trainer_config = PRISMConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy if args.eval_strategy else "no",
        eval_steps=args.eval_steps,
        run_name=args.run_name,
        report_to=args.report_to,
        time_epsilon=args.time_epsilon,
        loss_norm_type=args.loss_norm_type,
        prism_lambda=args.prism_lambda,
        prism_k=args.prism_k,
        prism_freeze_unmasking_head=args.prism_freeze_unmasking_head,
        loss_weight_type=args.loss_weight_type,
        remove_unused_columns=False,
        per_device_eval_batch_size=args.eval_batch_size if args.eval_batch_size is not None else 1024,
        bf16=args.bf16 if args.bf16 is not None else True,
        dataloader_num_workers=args.dataloader_num_workers if args.dataloader_num_workers is not None else 4,
    )

    print(f"  Building dataset via {task.name} adapter ({args.data_path})...")
    dataset, eval_dataset = task.build_datasets(
        tokenizer=tokenizer,
        data_path=args.data_path,
        seq_len=seq_len,
        eval_data_path=args.eval_data_path,
        limit_data=args.limit_data or 0,
        mask_until_token=args.mask_until_token,
        eval_fraction=args.eval_fraction or 0.0,
    )

    scheduler = LinearAlphaScheduler()
    
    # Wrap model + prism_head in a single nn.Module so the Trainer
    # includes both sets of parameters in the optimizer.
    class PRISMModelWrapper(nn.Module):
        def __init__(self, model, prism_head):
            super().__init__()
            self.model = model
            self.prism_head = prism_head
            self.config = model.config  # forwarded for Trainer compatibility

        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)

    wrapped_model = PRISMModelWrapper(model, prism_head)

    trainer = PRISMTrainer(
        model=wrapped_model,
        prism_head=prism_head,
        args=trainer_config,
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

    # Step 3: Train
    print(f"\n[4/4] Starting PRISM Fine-tuning...")
    trainer.train()

    # Save backbone checkpoint and PRISM head via the trainer's custom save path.
    trainer.save_model()

    print(f"\nDONE — PRISM checkpoint saved to '{args.output_dir}'")

if __name__ == "__main__":
    main()
