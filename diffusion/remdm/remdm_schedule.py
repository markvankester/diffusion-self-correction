"""
ReMDM confidence-based remasking schedule.

Implements the remasking schedule σ_t from:
  "Remasking Discrete Diffusion Models with Inference-Time Scaling"
  (Wang et al., 2026)

The key idea: at each reverse-diffusion step, unmasked tokens may be
*remasked* with probability proportional to the model's lack of confidence,
giving the model a chance to reconsider its earlier predictions.

This module provides:
  - compute_sigma_max: theoretical max remasking probability per step
  - compute_sigma: apply rescale / cap strategies
  - confidence_reweight: per-token σ via softmax(-ψ) weighting
  - compute_initial_confidence: forward pass to get fair ψ for all tokens
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# σ computation
# ---------------------------------------------------------------------------

def compute_sigma_max(alpha_s: float, alpha_t: float) -> float:
    """
    Maximum remasking probability for a step from time t → s.

    σ_max = (1 - α_s) / α_t

    This is the upper bound from the paper — setting σ higher would make
    the approximate posterior invalid.
    """
    if alpha_t <= 0:
        return 0.0
    return (1.0 - alpha_s) / alpha_t


def compute_sigma(
    alpha_s: float,
    alpha_t: float,
    eta_rescale: float = 1.0,
    eta_cap: float = 1.0,
) -> float:
    """
    Compute the remasking probability σ_t with rescale and cap.

    σ = min(η_cap, η_rescale · σ_max)

    Args:
        alpha_s: α at the target time s (= α((i-1)/T)).
        alpha_t: α at the current time t (= α(i/T)).
        eta_rescale: Multiplicative rescale factor ∈ [0, 1].  0 = no remasking.
        eta_cap: Hard upper bound on σ ∈ [0, 1].
    """
    sigma_max = compute_sigma_max(alpha_s, alpha_t)
    return min(eta_cap, eta_rescale * sigma_max)


# ---------------------------------------------------------------------------
# Per-token confidence reweighting
# ---------------------------------------------------------------------------

def confidence_reweight(
    sigma_base: float,
    psi_scores: torch.Tensor,
    mask_index: torch.Tensor,
    revisitable_region: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token remasking probability σ(ℓ) using confidence-based weighting.

    For each unmasked, revisitable token ℓ:
        η_conf(ℓ) = exp(-ψ(ℓ)) / Σ_{ℓ'} exp(-ψ(ℓ'))
        σ(ℓ) = η_conf(ℓ) · σ_base

    Tokens with low ψ (low confidence) get higher σ(ℓ).
    Masked tokens and non-revisitable tokens get σ(ℓ) = 0.

    Args:
        sigma_base: Scalar σ_t for this step (from compute_sigma).
        psi_scores: [B, T] confidence scores. ∞ for masked / clue tokens.
        mask_index: [B, T] bool — True where token is currently masked.
        revisitable_region: [B, T] bool — True where remasking is allowed.

    Returns:
        [B, T] per-token remasking probabilities, clamped to [0, 1].
    """
    B, T = psi_scores.shape
    device = psi_scores.device

    # Eligible for remasking: unmasked AND revisitable
    eligible = (~mask_index) & revisitable_region  # [B, T]

    if sigma_base <= 0 or not eligible.any():
        return torch.zeros(B, T, device=device)

    # Compute softmax(-ψ) over eligible tokens only.
    # Set ineligible positions to -inf so they get zero weight in softmax.
    neg_psi = -psi_scores.clone()
    neg_psi[~eligible] = -torch.inf

    # softmax across sequence dimension
    eta_conf = F.softmax(neg_psi, dim=1)  # [B, T]

    # Per-token σ(ℓ) = η_conf(ℓ) · σ_base
    sigma_per_token = eta_conf * sigma_base  # [B, T]

    # Zero out ineligible positions and clamp
    sigma_per_token[~eligible] = 0.0
    sigma_per_token = sigma_per_token.clamp(0.0, 1.0)

    return sigma_per_token


# ---------------------------------------------------------------------------
# Initial confidence via forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_initial_confidence(
    model,
    x_full: torch.Tensor,
    attention_mask: torch.Tensor,
    revisitable_region: torch.Tensor,
) -> torch.Tensor:
    """
    Run a forward pass on the fully-visible input to get per-token confidence.

    For each revisitable position ℓ, ψ(ℓ) = p_θ(x = z_ℓ | full context).
    For non-revisitable (clue) positions: ψ = ∞.

    This is used to provide fair initial confidence scores for tokens that
    were never predicted by the model during sampling (e.g., injected errors).

    Args:
        model: The pretrained MDLM backbone.
        x_full: [B, T] fully-visible token IDs (no masking).
        attention_mask: [B, T] padding mask.
        revisitable_region: [B, T] bool — True for positions that may be remasked.

    Returns:
        [B, T] confidence scores ψ.
        ψ = p_θ(actual token) for revisitable positions.
        ψ = ∞ for non-revisitable positions.
    """
    outputs = model(input_ids=x_full, attention_mask=attention_mask)
    logits = outputs.logits  # [B, T, V]

    p = F.softmax(logits, dim=-1)  # [B, T, V]

    # Gather the probability the model assigns to the actual token at each position
    psi = torch.gather(p, dim=-1, index=x_full.unsqueeze(-1)).squeeze(-1)  # [B, T]

    # Non-revisitable positions get ∞ (they are never remasked)
    psi[~revisitable_region] = float("inf")

    return psi
