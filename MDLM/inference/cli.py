from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from MDLM.utils import load_config_file

from MDLM.tasks import get_task_adapter

from .model_loading import load_model
from .runner import run_prompts


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "inference.toml"
TASK_CONFIG_PATHS = {
    "arithmetic": CONFIG_DIR / "inference_arithmetic.toml",
    "sudoku": CONFIG_DIR / "inference_sudoku.toml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MDLM inference script")

    parser.add_argument("--config", type=str, default=None, help="Path to a TOML config file")
    parser.add_argument("--task", type=str, choices=["arithmetic", "sudoku"], default=None)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="'cuda', 'cpu', or 'auto'")

    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default=None)
    parser.add_argument("--stochastic_transfer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--prism_eta", type=float, default=None)
    parser.add_argument("--prism_quality_threshold", type=float, default=None)
    parser.add_argument("--backplay_budget", type=int, default=None)
    parser.add_argument("--backplay_threshold", type=float, default=None)
    parser.add_argument("--backplay_stride", type=int, default=None)
    parser.add_argument("--backplay_block_buffer", type=int, default=None)
    parser.add_argument("--remdm_eta_rescale", type=float, default=None)
    parser.add_argument("--remdm_eta_cap", type=float, default=None)
    parser.add_argument("--remdm_ton", type=float, default=None)
    parser.add_argument("--remdm_toff", type=float, default=None)
    parser.add_argument("--remedi_threshold", type=float, default=None)

    parser.add_argument("--prompts", type=str, nargs="+", default=None)
    parser.add_argument("--prompt_file", type=str, default=None)

    parser.add_argument("--use_dataset", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_mode", type=str, default=None, choices=["first", "random"])
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--prompt_delimiter", type=str, default=None)

    parser.add_argument("--show_stats", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--infill", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show_steps", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--steps_log_dir", type=str, default=None, help="Root directory for inference logs")

    initial_args, _ = parser.parse_known_args()
    config_path, toml_config = _resolve_config(initial_args)
    parser.set_defaults(**{k: v for k, v in toml_config.items() if k != "prompts"})
    parser.set_defaults(config=str(config_path))
    args = parser.parse_args()

    task = get_task_adapter(args.task or "arithmetic")
    args.task = task.name
    _resolve_infill_mode(args)
    _resolve_prompts(args, task, toml_config)

    if args.checkpoint is None:
        args.checkpoint = "./checkpoints"
    if args.device is None:
        args.device = "auto"

    return args


def _resolve_infill_mode(args: argparse.Namespace) -> None:
    """Choose the default inference mode when the user/config did not specify one."""
    if args.infill is not None:
        return
    if args.task == "sudoku":
        args.infill = True
    elif args.task == "arithmetic":
        args.infill = args.remasking is not None
    else:
        args.infill = False


def _resolve_config(args: argparse.Namespace) -> tuple[Path, dict]:
    if args.config:
        config_path = Path(args.config)
        return config_path, load_config_file(config_path)

    if args.task in TASK_CONFIG_PATHS:
        config_path = TASK_CONFIG_PATHS[args.task]
        return config_path, load_config_file(config_path)

    selector_config = load_config_file(DEFAULT_CONFIG_PATH)
    task_name = selector_config.get("task")
    if task_name in TASK_CONFIG_PATHS:
        config_path = TASK_CONFIG_PATHS[task_name]
        return config_path, load_config_file(config_path)

    return DEFAULT_CONFIG_PATH, selector_config


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n[*] Device      : {device.type.upper()}")
    print(f"[*] Task        : {args.task}")
    print(f"[*] Checkpoint  : {args.checkpoint}")
    print(f"[*] Config File : {args.config}")

    if not os.path.exists(args.checkpoint):
        print(f"\nERROR: Checkpoint path '{args.checkpoint}' not found.")
        print(f"Please specify a valid --checkpoint or update {args.config}")
        return

    tokenizer, model, prism_head, backplay_head = load_model(args.checkpoint, device)
    print(f"[*] Model loaded - {sum(p.numel() for p in model.parameters()):,} parameters")
    if prism_head:
        print(f"[*] PRISM head loaded - {prism_head.num_parameters():,} parameters")
    if backplay_head:
        print(f"[*] BackPlay head loaded - {backplay_head.num_parameters():,} parameters")

    run_prompts(
        args.prompts,
        tokenizer,
        model,
        _sampler_kwargs(args),
        show_stats=getattr(args, "show_stats", False),
        infill=getattr(args, "infill", False),
        prism_head=prism_head,
        backplay_head=backplay_head,
        max_length=args.max_length or 20,
        task_name=args.task,
        solutions=getattr(args, "solutions", None),
        show_steps=getattr(args, "show_steps", False),
        steps_log_dir=getattr(args, "steps_log_dir", None),
        revisitable_regions=getattr(args, "revisitable_regions", None),
        injected_error_masks=getattr(args, "injected_error_masks", None),
        initial_confidence_texts=getattr(args, "initial_confidence_texts", None),
    )
    print()


def _resolve_prompts(args: argparse.Namespace, task, toml_config: dict) -> None:
    if args.prompts is not None:
        return

    if getattr(args, "use_dataset", False):
        path = args.dataset_path or task.default_inference_dataset_path
        mode = args.dataset_mode or "first"
        num = args.num_prompts if args.num_prompts is not None else 10
        delim = args.prompt_delimiter or task.default_prompt_delimiter
        num_print = "all" if num <= 0 else num
        print(f"[*] Loading {num_print} prompts from dataset ({path}, mode={mode}, task={args.task})...")
        args.prompts = task.load_dataset_prompts(path, mode, num, delim)
        args.solutions = getattr(task, "_solution_strings", None)
        args.revisitable_regions = getattr(task, "_revisitable_regions", None)
        args.injected_error_masks = getattr(task, "_injected_error_masks", None)
        args.initial_confidence_texts = getattr(task, "_initial_confidence_strings", None)

    if args.prompts:
        return

    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            args.prompts = [line.strip() for line in f if line.strip()]
    elif "prompts" in toml_config:
        args.prompts = toml_config["prompts"]
    else:
        args.prompts = task.default_prompts


def _sampler_kwargs(args: argparse.Namespace) -> dict:
    return {
        "steps": args.steps,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "block_size": args.block_size,
        "temperature": args.temperature,
        "remasking": args.remasking,
        "stochastic_transfer": args.stochastic_transfer,
        "cfg_scale": args.cfg_scale,
        "prism_eta": args.prism_eta,
        "prism_quality_threshold": args.prism_quality_threshold,
        "backplay_budget": args.backplay_budget,
        "backplay_threshold": args.backplay_threshold,
        "backplay_stride": args.backplay_stride,
        "backplay_block_buffer": args.backplay_block_buffer,
        "remdm_eta_rescale": args.remdm_eta_rescale,
        "remdm_eta_cap": args.remdm_eta_cap,
        "remdm_ton": args.remdm_ton,
        "remdm_toff": args.remdm_toff,
        "remedi_threshold": args.remedi_threshold,
    }
