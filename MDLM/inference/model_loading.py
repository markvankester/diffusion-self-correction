from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from backbones.llada.config import MDLMConfig
from backbones.llada.model import MDLMModelLM
from diffusion.backplay import BackPlayHead
from diffusion.prism import PRISMHead


def load_model(checkpoint: str, device: torch.device) -> tuple:
    """Load tokenizer, backbone model, and optional correction head."""
    checkpoint_path = Path(checkpoint)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)

    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing {config_path}. This PRISM checkpoint was saved without the backbone config; "
            "re-save it with the fixed PRISMTrainer.save_model path."
        )

    config = MDLMConfig.from_pretrained(checkpoint)
    architectures = getattr(config, "architectures", [])
    if "RemeDiUPMModelLM" in architectures:
        from backbones.llada.model import RemeDiUPMModelLM
        model = RemeDiUPMModelLM.from_pretrained(checkpoint, config=config)
    else:
        model = MDLMModelLM.from_pretrained(checkpoint, config=config)
    model.eval()
    model.to(device)

    prism_head = None
    prism_path = checkpoint_path / "prism_head.pt"
    if prism_path.exists():
        print(f"[*] Found PRISM head at {prism_path}")
        prism_config_path = checkpoint_path / "prism_head_config.json"
        if prism_config_path.exists():
            with open(prism_config_path, "r", encoding="utf-8") as f:
                prism_config = json.load(f)
        else:
            prism_config = {
                "d_model": model.config.d_model,
                "head_type": "attention",
                "n_heads": 4,
                "dropout": 0.0,
            }
        prism_head = PRISMHead.from_config_dict(prism_config)
        prism_head.load_state_dict(torch.load(prism_path, map_location=device, weights_only=True))
        prism_head.to(device)
        prism_head.eval()

    backplay_head = None
    backplay_path = checkpoint_path / "backplay_head.pt"
    if backplay_path.exists():
        print(f"[*] Found BackPlay head at {backplay_path}")
        backplay_config_path = checkpoint_path / "backplay_head_config.json"
        if backplay_config_path.exists():
            with open(backplay_config_path, "r", encoding="utf-8") as f:
                backplay_config = json.load(f)
        else:
            backplay_config = {
                "d_model": model.config.d_model,
                "n_layers": 2,
                "n_heads": 4,
                "dropout": 0.0,
                "hidden_state_index": -2,
            }
        backplay_head = BackPlayHead.from_config_dict(backplay_config)
        backplay_head.load_state_dict(torch.load(backplay_path, map_location=device, weights_only=True))
        backplay_head.to(device)
        backplay_head.eval()

    return tokenizer, model, prism_head, backplay_head
