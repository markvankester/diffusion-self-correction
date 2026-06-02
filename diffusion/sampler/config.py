from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class BaseSamplerOutput:
    """Output container for sampler results."""
    sequences: torch.Tensor
    histories: list[torch.Tensor] | None = None
    confidences: list[torch.Tensor] | None = None
    quality_scores: list[torch.Tensor] | None = None
    transfer_indices: list[torch.Tensor] | None = None
    remask_indices: list[torch.Tensor] | None = None
    x0_histories: list[torch.Tensor] | None = None


@dataclass
class BaseSamplerConfig:
    """Base configuration for all samplers."""
    return_dict: bool = False


@dataclass
class MDLMSamplerConfig(BaseSamplerConfig):
    """
    Configuration for MDLMSampler.

    Key parameters:
        steps: Number of reverse diffusion steps (more = higher quality, slower).
        temperature: Controls randomness. 0 = greedy, higher = more random.
        remasking: Strategy for choosing which tokens to unmask first.
            'low_confidence' = unmask highest-confidence predictions first.
            'random' = random selection order.
            'prism' = PRISM self-correction remasking.
            'backplay' = BackPlay error-head self-correction remasking.
            'remedi' = RemeDi unmasking policy stream remasking.
    """
    max_new_tokens: int | None = None
    max_length: int | None = None
    block_size: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    right_shift_logits: bool = False
    prism_eta: float = 0.0          # Probability of remasking a token (PRISM self-correction)
    prism_quality_threshold: float | None = None
    prism_single_block_infill: bool = True
    backplay_budget: int = 2
    backplay_threshold: float = 0.75
    backplay_stride: int = 4
    backplay_block_buffer: int = 4
    remdm_eta_rescale: float = 1.0  # Rescale factor for σ_max (ReMDM remasking intensity)
    remdm_eta_cap: float = 1.0      # Hard upper bound on σ (ReMDM)
    remdm_ton: float = 1.0          # Time t at which ReMDM turns ON (ReMDM-switch)
    remdm_toff: float = 0.0         # Time t at which ReMDM turns OFF (ReMDM-switch)
    remedi_threshold: float = float("inf") # Confidence threshold below which to remask tokens in RemeDi


def unpack_sampler_config(config: MDLMSamplerConfig, kwargs: dict) -> dict:
    """
    Extracts all config fields from MDLMSamplerConfig, allowing overrides via kwargs.
    """
    params = {}
    params["return_dict"] = kwargs.get("return_dict", config.return_dict)
    
    fields = [
        "max_new_tokens", "max_length", "block_size", "steps", "temperature",
        "remasking", "stochastic_transfer", "cfg_scale", "cfg_keep_tokens",
        "suppress_tokens", "begin_suppress_tokens", "right_shift_logits",
        "prism_eta", "prism_quality_threshold", "prism_single_block_infill",
        "backplay_budget", "backplay_threshold", "backplay_stride",
        "backplay_block_buffer", "remdm_eta_rescale", "remdm_eta_cap",
        "remdm_ton", "remdm_toff", "remedi_threshold"
    ]
    for field_name in fields:
        params[field_name] = kwargs.get(field_name, getattr(config, field_name))
    return params


DiffusionSamplerConfig = MDLMSamplerConfig
