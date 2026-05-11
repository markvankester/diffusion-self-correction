# Adapted from:
# https://github.com/ZHZisZZ/dllm

from __future__ import annotations

from dataclasses import dataclass

import torch

from .interfaces import TokenizerLike
from .schedules import BaseAlphaScheduler


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply Gumbel-Max noise for stochastic sampling from categorical distributions.

    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves
    perplexity but reduces generation quality. We use float64 for best quality.

    When temperature=0, returns logits unchanged (greedy decoding).
    Higher temperature → more randomness in token selection.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(
    mask_index: torch.Tensor,
    steps: int,
    scheduler: BaseAlphaScheduler,
    stochastic: bool = False,
) -> torch.Tensor:
    """
    Compute how many tokens to unmask at each diffusion step.

    For each sample in the batch, determines how many masked tokens should
    be revealed per step based on the reverse diffusion schedule.

    Args:
        mask_index: Boolean tensor [B, L] indicating which positions are masked.
        steps: Number of diffusion steps to run.
        scheduler: Alpha scheduler defining the masking schedule.
        stochastic: If True, sample from a binomial distribution (probabilistic);
            if False, use deterministic rounding of the expected count.

    Returns:
        Integer tensor [B, effective_steps] with number of tokens to unmask per step.
        Rows are compacted (zero-steps removed) and right-padded to the same length.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    )
    for i in range(mask_num.size(0)):
        for t, s, j in zip(range(steps, 0, -1), range(steps - 1, -1, -1), range(steps)):
            s /= steps
            t /= steps
            reverse_transfer_prob = 1 - scheduler.reverse_mask_prob(s=s, t=t)
            if not stochastic:
                x = mask_num[i, 0].to(torch.float64) * reverse_transfer_prob
                num_transfer_tokens[i, j] = torch.round(x).to(torch.int64)
            else:
                n = mask_num[i, 0].to(torch.float64)
                num_transfer_tokens[i, j] = (
                    torch.distributions.Binomial(n, reverse_transfer_prob)
                    .sample()
                    .to(torch.int64)
                )
            num_transfer_tokens[i, j] = torch.minimum(
                num_transfer_tokens[i, j], mask_num[i, 0]
            )
            mask_num[i, 0] -= num_transfer_tokens[i, j]
            if mask_num[i, 0].item() == 0:
                break

    # Compact: remove zero-steps per row, then right-pad to uniform length.
    # Because LLaDA is not conditioned on time, we can skip steps with no transfers.
    rows = []
    max_len = 0
    for i in range(num_transfer_tokens.size(0)):
        nonzero = num_transfer_tokens[i][num_transfer_tokens[i] > 0]
        rows.append(nonzero)
        max_len = max(max_len, nonzero.numel())
    padded_rows = []
    for r in rows:
        if r.numel() < max_len:
            pad = torch.zeros(max_len - r.numel(), dtype=r.dtype, device=r.device)
            r = torch.cat([r, pad])
        padded_rows.append(r)
    return torch.stack(padded_rows, dim=0)

def _decode_and_trim(tokenizer: TokenizerLike, gen_ids: list[int]) -> str:
    """Helper to decode tokens and truncate at the first EOS/EOT token."""
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eot_id = getattr(tokenizer, "eot_token_id", None)
    
    end = len(gen_ids)
    if eos_id is not None or eot_id is not None:
        for i in range(len(gen_ids)):
            if gen_ids[i] in (eos_id, eot_id):
                end = i
                break
                
    text = tokenizer.decode(gen_ids[:end], skip_special_tokens=True)
    eos = getattr(tokenizer, "eos_token", None)
    eot = getattr(tokenizer, "eot_token", None)
    if eos:
        text = text.split(eos)[0]
    if eot:
        text = text.split(eot)[0]
    return text


def sample_trim(tokenizer: TokenizerLike, seq_ids_list, input_ids_list) -> list[str]:
    """
    Return only the generated text, truncated at the first EOS **after** the prompt.

    Args:
        tokenizer: HF tokenizer with eos_token_id / pad_token_id.
        seq_ids_list: Full sequence token ids from the model (prompt + generation).
        input_ids_list: The prompt token ids that were fed into the model.

    Behavior:
        - Finds the first eos_token_id that occurs at or after len(input_ids).
        - Slices generation up to (but not including) that EOS.
        - Decodes only the generation span, skipping special/pad tokens.
    """
    sequences = []
    for seq_ids, input_ids in zip(seq_ids_list, input_ids_list):
        full = list(seq_ids)
        prompt = list(input_ids)

        # Skip left padding tokens (necessary for dream)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            while full and full[0] == pad_id:
                full.pop(0)

        start = len(prompt)
        sequences.append(_decode_and_trim(tokenizer, full[start:]))
    return sequences


def infill_trim(tokenizer: TokenizerLike, seq_ids_list, input_ids_list) -> list[str]:
    """
    Return only the generated text extracted from the originally masked region.
    """
    sequences = []
    for seq_ids, input_ids in zip(seq_ids_list, input_ids_list):
        full = torch.tensor(seq_ids)
        prompt = torch.tensor(input_ids)

        # Skip left padding tokens
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            while full.numel() and full[0].item() == pad_id:
                full = full[1:]

        masked_index = prompt == tokenizer.mask_token_id
        infill = full[masked_index].tolist()

        sequences.append(_decode_and_trim(tokenizer, infill))
    return sequences
