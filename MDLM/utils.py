from __future__ import annotations

import os
import tomllib
from pathlib import Path
from transformers import AutoTokenizer


def load_config_file(config_path: str | os.PathLike) -> dict:
    """Load and parse a TOML configuration file."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level TOML table.")
    return config


def build_tokenizer(tokenizer_id: str):
    """Load tokenizer from pretrained path with fallbacks for mask/pad tokens."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def resolve_config_path(
    config_arg: str | None,
    task: str | None,
    task_config_paths: dict[str, Path],
    default_config_path: Path,
) -> Path:
    """Resolve the TOML config file path from arguments, task defaults, or standard fallback."""
    if config_arg:
        return Path(config_arg)
    if task in task_config_paths:
        return task_config_paths[task]
    return default_config_path
