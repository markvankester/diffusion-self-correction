"""
run_inference.py
================
Load a trained MDLM checkpoint and sample predictions for a list of prompts.
"""

import argparse
import sys
import os
import tomllib
import json
from pathlib import Path

# Allow importing from the project root
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import PreTrainedTokenizerFast
from backbones.llada.config import MDLMConfig
from backbones.llada.model import MDLMModelLM
from diffusion.schedules import LinearAlphaScheduler
from diffusion.sampler import MDLMSampler, MDLMSamplerConfig
from diffusion.prism import PRISMHead
from diffusion import sample_trim, infill_trim
from MDLM.tasks import get_task_adapter


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "inference.toml"


def load_config_file(config_path: str | os.PathLike) -> dict:
    """Load evaluation parameters from a TOML file."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level TOML table.")
    return config


def load_model(checkpoint: str, device: torch.device) -> tuple:
    """Load tokenizer and model from a checkpoint directory."""
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
    config_path = Path(checkpoint) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing {config_path}. This PRISM checkpoint was saved without the backbone config; "
            "re-save it with the fixed PRISMTrainer.save_model path."
        )

    config = MDLMConfig.from_pretrained(checkpoint)
    model = MDLMModelLM.from_pretrained(checkpoint, config=config)
    model.eval()
    model.to(device)

    # Load PRISM head if it exists
    prism_head = None
    prism_path = Path(checkpoint) / "prism_head.pt"
    if prism_path.exists():
        print(f"[*] Found PRISM head at {prism_path}")
        prism_config_path = Path(checkpoint) / "prism_head_config.json"
        if prism_config_path.exists():
            with open(prism_config_path, "r", encoding="utf-8") as f:
                prism_config = json.load(f)
        else:
            prism_config = {"d_model": model.config.d_model, "head_type": "attention", "n_heads": 4, "dropout": 0.0}
        prism_head = PRISMHead.from_config_dict(prism_config)
        prism_head.load_state_dict(torch.load(prism_path, map_location=device, weights_only=True))
        prism_head.to(device)
        prism_head.eval()

    return tokenizer, model, prism_head


def run_prompts(prompts: list[str], tokenizer, model, sampler_kwargs: dict, show_stats: bool = False, infill: bool = False, prism_head: torch.nn.Module = None, max_length: int = 20):
    """Run MDLM sampling on a list of prompt strings and print results."""
    scheduler = LinearAlphaScheduler()
    sampler = MDLMSampler(model=model, tokenizer=tokenizer, scheduler=scheduler, prism_head=prism_head)

    # Filter out None values to use MDLMSamplerConfig defaults where applicable
    config_params = {k: v for k, v in sampler_kwargs.items() if v is not None}
    
    # Ensure block_size is set if not provided (defaulting to max_new_tokens for simple usage)
    if "block_size" not in config_params:
        config_params["block_size"] = config_params.get("max_new_tokens") or config_params.get("max_length", 16)

    sample_config = MDLMSamplerConfig(
        suppress_tokens=[
            token_id
            for token_id in (
                tokenizer.mask_token_id,
                tokenizer.bos_token_id,
                tokenizer.unk_token_id,
            )
            if token_id is not None
        ],
        return_dict=True,
        **config_params
    )

    print(f"\n{'='*60}")
    print(f"  RUNNING INFERENCE")
    for k, v in config_params.items():
        print(f"  [*] {k:15}: {v}")
    if show_stats:
        print(f"  [*] show_stats     : True")
    if infill:
        print(f"  [*] infill         : True")
    print(f"{'='*60}")

    for idx, prompt in enumerate(prompts, start=1):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        if infill:
            revisitable_region = None
            rhs_span = None
            # Arithmetic mode: if prompt already contains '=', make everything after
            # '=' editable by replacing RHS tokens with [MASK]. This keeps the query
            # (including '=') fixed while allowing PRISM remasking only on RHS.
            eq_token_id = tokenizer.convert_tokens_to_ids("=")
            if eq_token_id in prompt_ids:
                eq_idx = prompt_ids.index(eq_token_id)
                masked_prompt_ids = list(prompt_ids)
                has_rhs = eq_idx < (len(masked_prompt_ids) - 1)
                if has_rhs:
                    # Warm-start correction: keep provided RHS visible at step 0,
                    # and allow extra editable capacity up to max_length.
                    n_masks = max(0, max_length - len(masked_prompt_ids))
                    masked_prompt_ids = masked_prompt_ids + [tokenizer.mask_token_id] * n_masks
                    rhs_span = (eq_idx + 1, len(masked_prompt_ids))
                else:
                    # Query-only prompt like "123+45=": create an editable RHS region.
                    n_masks = max(0, max_length - len(masked_prompt_ids))
                    masked_prompt_ids = masked_prompt_ids + [tokenizer.mask_token_id] * n_masks
                    rhs_span = (eq_idx + 1, len(masked_prompt_ids))
                revisitable_region = [False] * len(masked_prompt_ids)
                for pos in range(rhs_span[0], rhs_span[1]):
                    revisitable_region[pos] = True
                display_prompt_ids = prompt_ids[: eq_idx + 1]
                display_prefix = tokenizer.decode(display_prompt_ids, skip_special_tokens=True)
            else:
                # Fallback: no '=' present, behave like standard infill and append masks.
                n_masks = max(0, max_length - len(prompt_ids))
                masked_prompt_ids = prompt_ids + [tokenizer.mask_token_id] * n_masks
                revisitable_region = [False] * len(masked_prompt_ids)
                for pos in range(len(prompt_ids), len(masked_prompt_ids)):
                    revisitable_region[pos] = True
                rhs_span = (len(prompt_ids), len(masked_prompt_ids))
                display_prompt_ids = prompt_ids
                display_prefix = prompt

            output = sampler.infill(
                [masked_prompt_ids],
                config=sample_config,
                revisitable_region=[revisitable_region],
            )
            generated_ids = output.sequences[0].tolist()
            # For arithmetic infill, decode the full editable RHS span so both
            # warm-start digits and newly infilled tail are visible.
            start, end = rhs_span
            answer = tokenizer.decode(generated_ids[start:end], skip_special_tokens=True)

            print(f"\n  [{idx}] [Original] : {prompt}")
            print(f"      [Infilled] : {display_prefix}{answer.strip()}")
        else:
            output = sampler.sample([prompt_ids], config=sample_config)
            generated_ids = output.sequences[0].tolist()
            answer = sample_trim(tokenizer, [generated_ids], [prompt_ids])[0]
            display_prompt_ids = prompt_ids  # stats start after the query
            print(f"\n  [{idx}] {prompt}{answer.strip()}")

        if show_stats and output.histories is not None:
            print(f"\n    --- Inference Stats for Prompt [{idx}] ---")
            for step_idx in range(len(output.histories) - 1):
                # We start from step 1 (histories[0] is the initial fully-masked state)
                curr_seq_tensor = output.histories[step_idx + 1][0]
                curr_seq = curr_seq_tensor.tolist()
                step_confidence = (
                    output.confidences[step_idx][0]
                    if output.confidences is not None and step_idx < len(output.confidences)
                    else None
                )

                if output.confidences is not None and output.transfer_indices is not None and step_idx < len(output.confidences):
                    quality = (
                        output.quality_scores[step_idx][0]
                        if output.quality_scores is not None
                        and step_idx < len(output.quality_scores)
                        and output.quality_scores[step_idx] is not None
                        else None
                    )
                    trans_idx = output.transfer_indices[step_idx][0]
                    unmasked_this_step = trans_idx.nonzero(as_tuple=True)[0].tolist()
                    if output.remask_indices is not None and step_idx < len(output.remask_indices):
                        remask_idx = output.remask_indices[step_idx][0]
                        remasked_this_step = remask_idx.nonzero(as_tuple=True)[0].tolist()
                    else:
                        remasked_this_step = []
                else:
                    quality = None
                    unmasked_this_step = []
                    remasked_this_step = []

                visual_seq = []
                for pos in range(len(display_prompt_ids), len(curr_seq)):
                    token_id = curr_seq[pos]
                    token_str = tokenizer.convert_ids_to_tokens(token_id)
                    
                    if pos in remasked_this_step:
                        prev_token_id = output.histories[step_idx][0][pos].item()
                        prev_token_str = tokenizer.convert_ids_to_tokens(prev_token_id)
                        if quality is not None:
                            visual_seq.append(f"\033[91m[{prev_token_str} -> MASK (q={quality[pos].item():.2f})]\033[0m")
                        else:
                            visual_seq.append(f"\033[91m[{prev_token_str} -> MASK]\033[0m")
                    elif pos in unmasked_this_step and token_str != "[MASK]":
                        p_val = (
                            step_confidence[pos].item()
                            if step_confidence is not None
                            else float("nan")
                        )
                        rendered = f"{token_str}(p={p_val:.2f})"
                        visual_seq.append(f"\033[92m{rendered}\033[0m")
                    elif token_str == "[MASK]":
                        visual_seq.append(f"\033[90m[MASK]\033[0m")
                    else:
                        if quality is not None:
                            visual_seq.append(f"{token_str}(q={quality[pos].item():.2f})")
                        else:
                            visual_seq.append(token_str)
                        
                curr_str = " ".join(visual_seq)
                print(f"    Step {step_idx + 1:02d} | {curr_str}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MDLM inference script")

    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH),
                        help="Path to a TOML config file")
    parser.add_argument("--task", type=str, choices=["arithmetic", "sudoku"], default=None,
                        help="Task adapter used for dataset prompts and defaults.")

    # Paths & Device
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the checkpoint folder")
    parser.add_argument("--device",     type=str, default=None,
                        help="'cuda', 'cpu', or 'auto'")

    # Sampler Hyperparameters
    parser.add_argument("--steps",          type=int,   default=None)
    parser.add_argument("--max_length",     type=int,   default=None)
    parser.add_argument("--max_new_tokens", type=int,   default=None)
    parser.add_argument("--block_size",     type=int,   default=None)
    parser.add_argument("--temperature",    type=float, default=0.0)
    parser.add_argument("--remasking",      type=str,   default=None)
    parser.add_argument("--stochastic_transfer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cfg_scale",      type=float, default=0.0)
    parser.add_argument("--prism_eta",      type=float, default=None,
                        help="Remasking rate for PRISM strategy (0 to 1).")
    parser.add_argument("--prism_quality_threshold", type=float, default=None,
                        help="Only remask tokens with PRISM quality below this threshold.")


    # Prompts
    parser.add_argument("--prompts",     type=str,   nargs="+", default=None,
                        help="List of prompts to run")
    parser.add_argument("--prompt_file", type=str,   default=None,
                        help="Path to a text file with one prompt per line")
    
    # Dataset Selection
    parser.add_argument("--use_dataset", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dataset_path", type=str,  default=None)
    parser.add_argument("--dataset_mode", type=str,  default=None, choices=["first", "random"])
    parser.add_argument("--num_prompts",  type=int,  default=None)
    parser.add_argument("--prompt_delimiter", type=str, default=None)

    # Misc
    parser.add_argument("--show_stats",  action=argparse.BooleanOptionalAction, default=None,
                        help="Whether to show step-by-step inference statistics.")
    parser.add_argument("--infill",      action=argparse.BooleanOptionalAction, default=True,
                        help="Use the infilling sampling method (requires [MASK] tokens in prompt).")

    # 1. Load TOML first
    initial_args, _ = parser.parse_known_args()
    toml_config = load_config_file(initial_args.config)
    
    # 2. Set defaults from TOML for normal arguments
    # Filter out 'prompts' as it's handled specially
    toml_defaults = {k: v for k, v in toml_config.items() if k != "prompts"}
    parser.set_defaults(**toml_defaults)
    
    # 3. Parse all arguments (CLI overrides TOML)
    args = parser.parse_args()
    
    # Handle prompts priority: CLI > Dataset > CLI prompt_file > TOML > default list
    task = get_task_adapter(args.task or "arithmetic")
    args.task = task.name

    if args.prompts is None:
        if getattr(args, "use_dataset", False):
            path = args.dataset_path or task.default_inference_dataset_path
            mode = args.dataset_mode or "first"
            num = args.num_prompts or 10
            delim = args.prompt_delimiter or task.default_prompt_delimiter
            print(f"[*] Loading {num} prompts from dataset ({path}, mode={mode}, task={args.task})...")
            args.prompts = task.load_dataset_prompts(path, mode, num, delim)
        
        # Fallbacks if dataset failed or wasn't used
        if not args.prompts:
            if args.prompt_file and os.path.exists(args.prompt_file):
                with open(args.prompt_file, "r") as f:
                    args.prompts = [line.strip() for line in f if line.strip()]
            elif "prompts" in toml_config:
                args.prompts = toml_config["prompts"]
            else:
                args.prompts = task.default_prompts
            
    # Final default values for critical fields if neither TOML nor CLI provided them
    if args.checkpoint is None: args.checkpoint = "./checkpoints"
    if args.device is None:     args.device = "auto"

    return args


def main():
    args = parse_args()

    # Determine device
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
        print("Please specify a valid --checkpoint or update inference.toml")
        return

    tokenizer, model, prism_head = load_model(args.checkpoint, device)
    print(f"[*] Model loaded — {sum(p.numel() for p in model.parameters()):,} parameters")
    if prism_head:
        print(f"[*] PRISM head loaded — {prism_head.num_parameters():,} parameters")

    sampler_kwargs = {
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
    }

    show_stats = getattr(args, "show_stats", False)
    infill = getattr(args, "infill", False)
    max_length = args.max_length or 20

    run_prompts(args.prompts, tokenizer, model, sampler_kwargs, show_stats=show_stats, infill=infill, prism_head=prism_head, max_length=max_length)
    print()


if __name__ == "__main__":
    main()
