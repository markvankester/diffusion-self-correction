"""
Inspect how text examples are tokenized and collated for MDLM training.
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from rich.console import Console
from rich.panel import Panel

from data.processing.tokenization import tokenize_and_pad
from MDLM.tasks import get_task_adapter

console = Console()

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "train.toml"


def load_config_file(config_path: str | os.PathLike) -> dict:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level TOML table.")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect tokenization and padding")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH),
                        help="Path to a TOML config file")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--tokenizer_id", type=str, default=None)
    parser.add_argument("--task", type=str, choices=["arithmetic", "sudoku"], default="sudoku")
    parser.add_argument("--seq_len", type=int, default=81)
    parser.add_argument("--examples", type=int, default=3,
                        help="How many dataset rows to inspect")
    parser.add_argument("--offset", type=int, default=0,
                        help="Dataset row offset")
    parser.add_argument("--mask_until_token", type=str, default=None,
                        help="Optional string up to which tokens will be masked")

    initial_args, _ = parser.parse_known_args()
    parser.set_defaults(**load_config_file(initial_args.config))
    parser.set_defaults(config=initial_args.config)
    args = parser.parse_args()

    task = get_task_adapter(args.task or "arithmetic")
    args.task = task.name
    if args.data_path is None:
        args.data_path = task.default_data_path
    if args.tokenizer_id is None:
        args.tokenizer_id = task.default_tokenizer_path
    if args.seq_len is None:
        args.seq_len = 16
    return args


# ── Rendering helpers ──────────────────────────────────────────────────


def _token_str(tokenizer, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    return "<space>" if token == " " else token


def render_sequence(tokenizer, ids: list[int]) -> str:
    parts = []
    for token_id in ids:
        tok = _token_str(tokenizer, token_id)
        if token_id == tokenizer.pad_token_id:
            parts.append(f"[dim]{tok}[/dim]")
        else:
            parts.append(f"[green]{tok}[/green]")
    return " ".join(parts)


def render_labels(tokenizer, ids: list[int], label_ids: list[int]) -> str:
    """Render label row aligned with tokens. Maskable = cyan ID, ignored = dim dot."""
    parts = []
    for token_id, label in zip(ids, label_ids):
        tok = _token_str(tokenizer, token_id)
        width = max(len(tok), 1)
        if label == -100:
            parts.append(f"[dim]{'·':^{width}}[/dim]")
        else:
            parts.append(f"[cyan]{str(label):^{width}}[/cyan]")
    return " ".join(parts)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    task = get_task_adapter(args.task)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    examples = task.load_examples(args.data_path, offset=args.offset, limit=args.examples)

    # ── Tokenize ──────────────────────────────────────────────────────
    batch = tokenize_and_pad(
        {"text": examples},
        tokenizer=tokenizer,
        text_field="text",
        seq_length=args.seq_len,
        insert_eos=args.task != "sudoku",
        mask_until_token=args.mask_until_token,
    )

    # ── Collate (pad + tensorise) ─────────────────────────────────────
    features = [
        {"input_ids": batch["input_ids"][i], "labels": batch["labels"][i]}
        for i in range(len(batch["input_ids"]))
    ]
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding="max_length",
        max_length=args.seq_len,
        return_tensors="pt",
        label_pad_token_id=tokenizer.pad_token_id,
    )
    padded = collator(features)
    input_ids = padded["input_ids"]
    labels = padded["labels"]
    attention_mask = padded["attention_mask"]

    # ── Settings panel ────────────────────────────────────────────────
    console.print(Panel.fit(
        f"[bold]Config[/bold]\n"
        f"  task          : {args.task}\n"
        f"  data_path     : {args.data_path}\n"
        f"  tokenizer_id  : {args.tokenizer_id}\n"
        f"  seq_len       : {args.seq_len}\n"
        f"  examples      : {len(examples)}\n"
        f"  mask_delimiter: '{args.mask_until_token}'",
        title="Settings", border_style="cyan"
    ))

    if input_ids.numel() == 0:
        console.print(
            Panel.fit(
                "[yellow]No sequences were produced.[/yellow]\n"
                "Try a smaller `--seq_len` or a larger `--examples` count.",
                title="Empty Output",
                border_style="yellow",
            )
        )
        return

    # ── Print each sequence ───────────────────────────────────────────
    for i in range(input_ids.size(0)):
        ids = input_ids[i]
        label_ids = labels[i]
        attn = attention_mask[i]
        valid_len = int(attn.sum().item())
        maskable = int((label_ids != -100).sum().item())

        console.print(f"\n[bold yellow]{'=' * 72}[/bold yellow]")

        text = examples[i] if i < len(examples) else "?"
        console.print(f"[bold bright_blue]Example {args.offset + i}[/bold bright_blue]")
        console.print(f"  raw text       : [bold white]{text}[/bold white]")
        console.print(f"  valid tokens   : [cyan]{valid_len}[/cyan]")
        console.print(f"  maskable       : [cyan]{maskable}[/cyan]")
        for label, value in task.describe_example(text):
            console.print(f"  {label:<14}: [yellow]{value}[/yellow]")

        console.print(f"\n  tokens : {render_sequence(tokenizer, ids.tolist())}")
        console.print(f"  labels : {render_labels(tokenizer, ids.tolist(), label_ids.tolist())}\n")


if __name__ == "__main__":
    main()
