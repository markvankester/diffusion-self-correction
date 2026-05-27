"""Train a frozen-backbone BackPlay correction head."""

from pathlib import Path
import argparse
import os
import sys
import tomllib

import torch
import torch.nn as nn
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backbones.llada.config import MDLMConfig
from backbones.llada.model import MDLMModelLM
from data.processing.collators import NoAttentionMaskWrapper
from diffusion.backplay import BackPlayConfig, BackPlayHead, BackPlayTrainer
from diffusion.schedules import LinearAlphaScheduler
from MDLM.tasks import get_task_adapter


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "backplay_arithmetic.toml"
TASK_CONFIG_PATHS = {
    "arithmetic": CONFIG_DIR / "backplay_arithmetic.toml",
    "sudoku": CONFIG_DIR / "backplay_sudoku.toml",
}


def load_config_file(config_path: str | os.PathLike) -> dict:
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level TOML table.")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BackPlay correction head")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--task", type=str, choices=["arithmetic", "sudoku"], default=None)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--limit_data", type=int, default=None)
    parser.add_argument("--mask_until_token", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=None)

    parser.add_argument("--backplay_head_layers", type=int, default=2)
    parser.add_argument("--backplay_head_n_heads", type=int, default=4)
    parser.add_argument("--backplay_head_dropout", type=float, default=0.0)
    parser.add_argument("--backplay_head_ffn_dim", type=int, default=0)
    parser.add_argument("--backplay_head_type", type=str, default="attention")
    parser.add_argument("--backplay_hidden_state_index", type=int, default=-2)
    parser.add_argument("--backplay_delta_t", type=float, default=0.0625)
    parser.add_argument("--backplay_loss_scope", type=str, default="non_mask")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=50000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_strategy", type=str, default="steps")
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--eval_fraction", type=float, default=0.0)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--time_epsilon", type=float, default=1e-3)
    parser.add_argument("--loss_norm_type", type=str, default="token")
    parser.add_argument("--loss_weight_type", type=str, default="scheduler")

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


def main() -> None:
    args = parse_args()
    task = get_task_adapter(args.task or "arithmetic")
    args.task = task.name
    if args.data_path is None:
        args.data_path = task.default_data_path

    print(f"\n[1/4] Loading frozen MDLM for task={args.task} from {args.checkpoint}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.checkpoint)
    checkpoint_config = MDLMConfig.from_pretrained(args.checkpoint)
    model = MDLMModelLM.from_pretrained(args.checkpoint, config=checkpoint_config)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    seq_len = args.seq_len if args.seq_len is not None else checkpoint_config.max_sequence_length

    training_args_path = Path(args.checkpoint) / "training_args.bin"
    if training_args_path.exists():
        print("  [*] Syncing diffusion loss constraints from checkpoint...")
        try:
            pretrained_args = torch.load(training_args_path, map_location="cpu", weights_only=False)
            for name in ("time_epsilon", "loss_weight_type", "loss_norm_type"):
                if hasattr(pretrained_args, name) and getattr(args, name, None) is None:
                    setattr(args, name, getattr(pretrained_args, name))
        except Exception as exc:
            print(f"  [!] Failed to load training_args.bin for syncing: {exc}")

    print("[2/4] Attaching BackPlay head...")
    backplay_head = BackPlayHead(
        d_model=model.config.d_model,
        n_layers=args.backplay_head_layers,
        n_heads=args.backplay_head_n_heads,
        dropout=args.backplay_head_dropout,
        dim_feedforward=args.backplay_head_ffn_dim if args.backplay_head_ffn_dim else None,
        hidden_state_index=args.backplay_hidden_state_index,
        head_type=args.backplay_head_type if getattr(args, "backplay_head_type", None) else "attention",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    backplay_head.to(device)
    print(f"  BackPlay head parameters: {backplay_head.num_parameters():,}")

    print("[3/4] Building dataset and trainer...")
    dataset, eval_dataset = task.build_datasets(
        tokenizer=tokenizer,
        data_path=args.data_path,
        seq_len=seq_len,
        eval_data_path=args.eval_data_path,
        limit_data=args.limit_data or 0,
        mask_until_token=args.mask_until_token,
        eval_fraction=args.eval_fraction or 0.0,
    )

    trainer_config = BackPlayConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size if args.eval_batch_size is not None else 1024,
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
        loss_weight_type=args.loss_weight_type,
        remove_unused_columns=False,
        bf16=args.bf16 if args.bf16 is not None else True,
        dataloader_num_workers=args.dataloader_num_workers if args.dataloader_num_workers is not None else 4,
        backplay_delta_t=args.backplay_delta_t,
        backplay_loss_scope=args.backplay_loss_scope,
    )

    class BackPlayModelWrapper(nn.Module):
        def __init__(self, model, backplay_head):
            super().__init__()
            self.model = model
            self.backplay_head = backplay_head
            self.config = model.config

        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)

    trainer = BackPlayTrainer(
        model=BackPlayModelWrapper(model, backplay_head),
        backplay_head=backplay_head,
        args=trainer_config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        scheduler=LinearAlphaScheduler(),
        data_collator=NoAttentionMaskWrapper(
            DataCollatorForSeq2Seq(
                tokenizer,
                return_tensors="pt",
                padding=True,
                label_pad_token_id=tokenizer.pad_token_id,
            )
        ),
    )

    print("\n[4/4] Starting BackPlay head training...")
    trainer.train()
    trainer.save_model()
    print(f"\nDONE - BackPlay checkpoint saved to '{args.output_dir}'")


if __name__ == "__main__":
    main()
