# Adapted from:
# https://github.com/ZHZisZZ/dllm

"""
Reverse-diffusion sampler for Masked Diffusion Language Models (MDLM).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .interfaces import DiffusionModelLike, TokenizerLike
from .schedules import BaseAlphaScheduler, LinearAlphaScheduler


from .utils import add_gumbel_noise, get_num_transfer_tokens


# ---------------------------------------------------------------------------
# Output and Config dataclasses
# ---------------------------------------------------------------------------

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
    """
    max_new_tokens: int = None
    max_length: int = None
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


# ---------------------------------------------------------------------------
# Internal context dataclass for _run_diffusion_step
# ---------------------------------------------------------------------------

@dataclass
class _StepContext:
    """
    Carries the per-block/per-step context that differs between sample() and infill().

    Attributes:
        x: Current token sequence [B, T] — mutated in-place.
        attention_mask: Padding mask [B, T].
        unmasked_index: Boolean mask [B, T] of tokens that were never masked (for CFG).
        revisitable_region: Boolean mask [B, T] of positions that PRISM may remask.
            In sample() this is the generation region; in infill() it is original_mask_region.
        block_clamp_fn: Callable(x0_p, j) -> None that zeroes out confidence scores
            outside the current block's valid region for sample j.
        num_transfer_tokens: [B, effective_steps] int64 tensor, or None for PRISM mode.
        step_idx: Current step index within the block (used to index num_transfer_tokens).
        total_steps: Total steps in this block (used to compute PRISM time fractions).
        is_final_step: True on the very last step of the very last block.
        B: Batch size.
        mask_id: Token ID of the [MASK] token.
    """
    x: torch.Tensor
    attention_mask: torch.Tensor
    unmasked_index: torch.Tensor
    revisitable_region: torch.Tensor
    block_clamp_fn: object  # Callable[[torch.Tensor, int], None]
    num_transfer_tokens: torch.Tensor | None
    step_idx: int
    total_steps: int
    is_final_step: bool
    B: int
    mask_id: int


# ---------------------------------------------------------------------------
# MDLMSampler
# ---------------------------------------------------------------------------

@dataclass
class MDLMSampler:
    """
    Masked Diffusion Language Model sampler.

    Performs reverse diffusion: starts from a fully masked sequence and
    iteratively unmasks tokens over multiple steps, guided by the model's
    predictions and confidence scores.

    Attributes:
        model: The trained LLaDA/MDLM model.
        tokenizer: Tokenizer with mask_token_id, eos_token_id, bos_token_id.
        scheduler: Alpha scheduler for the diffusion process.
        prism_head: Optional PRISMHead for quality-guided remasking.
    """
    model: DiffusionModelLike
    tokenizer: TokenizerLike
    scheduler: BaseAlphaScheduler | None = None
    prism_head: nn.Module | None = None

    def __post_init__(self):
        if self.scheduler is None:
            self.scheduler = LinearAlphaScheduler()

    # ------------------------------------------------------------------
    # Shared inner loop
    # ------------------------------------------------------------------

    def _run_diffusion_step(
        self,
        ctx: _StepContext,
        cfg_scale: float,
        suppress_tokens: list[int] | None,
        begin_suppress_tokens: list[int] | None,
        right_shift_logits: bool,
        temperature: float,
        remasking: str,
        stochastic_transfer: bool,
        prism_eta: float,
        prism_quality_threshold: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Execute one reverse-diffusion step, shared by sample() and infill().

        Performs:
          1. Forward pass (with optional CFG) to get logits (and hidden states for PRISM).
          2. Apply token suppression / logit shift.
          3. Argmax + Gumbel noise → candidate tokens x0.
          4. Compute per-position confidence scores.
          5. Select which masked positions to commit (and optionally remask via PRISM).
          6. Update ctx.x in-place.

        Returns (for history tracking):
            confidence, transfer_index, remask_index, x0, quality_scores
        """
        x = ctx.x
        attention_mask = ctx.attention_mask
        mask_id = ctx.mask_id
        B = ctx.B

        mask_index = (x == mask_id)
        remask_index = torch.zeros_like(x, dtype=torch.bool)
        quality_scores = None
        hidden_states = None

        # ---- 1. Forward pass ------------------------------------------------
        if cfg_scale > 0.0:
            un_x = x.clone()
            un_x[ctx.unmasked_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = self.model(
                x_, attention_mask=attention_mask.repeat(2, 1)
            ).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            # PRISM still needs hidden states — do a second pass without CFG
            if remasking == "prism" and self.prism_head is not None and prism_eta > 0.0:
                hidden_states = self.model(
                    input_ids=x,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                ).hidden_states[-1]
        else:
            output_hidden_states = (remasking == "prism")
            outputs = self.model(
                input_ids=x,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
            )
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1] if output_hidden_states else None

        # ---- 2. Token suppression / logit shift ----------------------------
        if suppress_tokens is not None and len(suppress_tokens) > 0:
            suppress_ids = torch.as_tensor(suppress_tokens, dtype=torch.long, device=logits.device)
            logits[:, :, suppress_ids] = -torch.inf

        if right_shift_logits:
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        # NOTE: begin_suppress_tokens suppresses these token IDs at all positions,
        # not just the first. Scope to logits[:, 0, :] for true first-token suppression.
        if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
            begin_suppress_ids = torch.as_tensor(begin_suppress_tokens, dtype=torch.long, device=logits.device)
            logits[:, :, begin_suppress_ids] = -torch.inf

        # ---- 3. Candidate tokens -------------------------------------------
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        # ---- 4. Per-position confidence ------------------------------------
        if remasking in ("low_confidence", "prism"):
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        elif remasking == "random":
            x0_p = torch.rand_like(x0, dtype=torch.float32)
        else:
            raise NotImplementedError(f"Unknown remasking strategy: {remasking!r}")

        # Clamp confidence to the current block's valid region
        for j in range(B):
            ctx.block_clamp_fn(x0_p, j)

        # Confidence is only meaningful at currently-masked positions
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x.device))

        transfer_index = torch.zeros_like(x, dtype=torch.bool)

        # ---- 5a. PRISM remasking commit ------------------------------------
        if remasking == "prism":
            if not ctx.is_final_step and self.prism_head is not None and prism_eta > 0.0:
                if hidden_states is None:
                    raise RuntimeError(
                        "PRISM remasking requires hidden states from the backbone"
                    )
                is_clean = (x != mask_id) & attention_mask.bool() & ctx.revisitable_region
                if is_clean.any():
                    quality_scores = self.prism_head(hidden_states, attention_mask=attention_mask)
                    n_clean = is_clean.sum(dim=1)
                    n_remask = torch.distributions.Binomial(n_clean.float(), prism_eta).sample().long()
                    for j in range(B):
                        k_remask = int(n_remask[j].item())
                        if k_remask > 0:
                            q_j = quality_scores[j].clone()
                            eligible_mask = is_clean[j]
                            if prism_quality_threshold is not None:
                                eligible_mask = eligible_mask & (q_j < prism_quality_threshold)
                            q_j[~eligible_mask] = 2.0  # push ineligible out of topk range
                            n_eligible = int(eligible_mask.sum().item())
                            if n_eligible > 0:
                                k_remask = min(k_remask, n_eligible)
                                _, to_remask_idx = torch.topk(-q_j, k=k_remask)
                                remask_index[j, to_remask_idx] = True

            current_mask_region = mask_index & ctx.revisitable_region
            step_t = (ctx.total_steps - ctx.step_idx) / ctx.total_steps
            prev_t = (ctx.total_steps - ctx.step_idx - 1) / ctx.total_steps
            reverse_transfer_prob = 1 - self.scheduler.reverse_mask_prob(s=prev_t, t=step_t)

            for j in range(B):
                n_masked = int(current_mask_region[j].sum().item())
                if n_masked == 0:
                    continue
                if stochastic_transfer:
                    base_unmask = int(
                        torch.distributions.Binomial(
                            torch.tensor(float(n_masked), device=x.device),
                            torch.tensor(float(reverse_transfer_prob), device=x.device),
                        ).sample().item()
                    )
                else:
                    base_unmask = int(round(n_masked * float(reverse_transfer_prob)))

                # Extra slots freed up by incoming remasks
                extra_unmask = int(remask_index[j].sum().item())
                k_commit = min(n_masked, base_unmask + extra_unmask)
                if k_commit > 0:
                    _, select_idx = torch.topk(confidence[j], k=k_commit)
                    transfer_index[j, select_idx] = True

            x[transfer_index] = x0[transfer_index]
            x[remask_index] = mask_id

        # ---- 5b. Standard (low_confidence / random) commit -----------------
        else:
            assert ctx.num_transfer_tokens is not None
            for j in range(B):
                k = int(ctx.num_transfer_tokens[j, ctx.step_idx].item())
                if k > 0:
                    _, select_idx = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_idx] = True
            x[transfer_index] = x0[transfer_index]

        return confidence, transfer_index, remask_index, x0, quality_scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: MDLMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Generate text by appending mask tokens to prompts and iteratively unmasking.

        This is the standard "left-to-right style" generation: the prompt is kept
        fixed, and new tokens are generated in the masked region to the right.

        Args:
            inputs: List of prompt token tensors or lists of token IDs.
            config: Sampler configuration, or None for defaults.
            **kwargs: Override specific config parameters.

        Returns:
            BaseSamplerOutput (if return_dict=True) or raw tensor of token IDs.
        """
        if config is None:
            config = MDLMSamplerConfig()

        # Pull args from config, allow kwargs to override
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get("stochastic_transfer", config.stochastic_transfer)
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get("begin_suppress_tokens", config.begin_suppress_tokens)
        prism_eta = kwargs.get("prism_eta", config.prism_eta)
        prism_quality_threshold = kwargs.get("prism_quality_threshold", config.prism_quality_threshold)

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # Handle empty prompts when using right-shifted logits
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        if remasking == "prism":
            # PRISM remasking needs the whole generated region to stay revisitable.
            block_size = max_new_tokens

        B = len(inputs)
        T = max_length

        # Initialize canvas: EOS padding, then copy prompts and append mask tail
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = mask_id
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        # Track originally unmasked tokens for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # Generation region — PRISM may only remask tokens within this zone
        gen_region_mask = torch.zeros((B, T), dtype=torch.bool, device=x.device)
        for j in range(B):
            gen_region_mask[j, prompt_lens[j] : prompt_lens[j] + max_new_tokens] = True

        # Block scheduling
        num_blocks = math.ceil(max_new_tokens / block_size)
        base_steps = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None
        confidences_history = [] if return_dict else None
        quality_history = [] if return_dict else None
        transfer_history = [] if return_dict else None
        remask_history = [] if return_dict else None
        x0_histories = [] if return_dict else None

        for b in range(num_blocks):
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )
            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = (x[j, start:end] == mask_id)

            if remasking == "prism":
                effective_steps = base_steps
                num_transfer_tokens = None
            else:
                num_transfer_tokens = get_num_transfer_tokens(
                    mask_index=block_mask_index,
                    steps=base_steps,
                    scheduler=self.scheduler,
                    stochastic=stochastic_transfer,
                )
                effective_steps = num_transfer_tokens.size(1)

            for i in range(effective_steps):
                is_final_step = (b == num_blocks - 1) and (i == effective_steps - 1)

                # Block-boundary clamping: confidence is -inf beyond the right edge
                # of the current block for each sequence.
                def _clamp(x0_p: torch.Tensor, j: int, _b: int = b, _pl: list = prompt_lens) -> None:
                    x0_p[j, _pl[j] + (_b + 1) * block_size :] = -np.inf

                ctx = _StepContext(
                    x=x,
                    attention_mask=attention_mask,
                    unmasked_index=unmasked_index,
                    revisitable_region=gen_region_mask,
                    block_clamp_fn=_clamp,
                    num_transfer_tokens=num_transfer_tokens,
                    step_idx=i,
                    total_steps=effective_steps,
                    is_final_step=is_final_step,
                    B=B,
                    mask_id=mask_id,
                )

                confidence, transfer_index, remask_index, x0, quality_scores = (
                    self._run_diffusion_step(
                        ctx=ctx,
                        cfg_scale=cfg_scale,
                        suppress_tokens=suppress_tokens,
                        begin_suppress_tokens=begin_suppress_tokens,
                        right_shift_logits=right_shift_logits,
                        temperature=temperature,
                        remasking=remasking,
                        stochastic_transfer=stochastic_transfer,
                        prism_eta=prism_eta,
                        prism_quality_threshold=prism_quality_threshold,
                    )
                )

                if histories is not None:
                    histories.append(x.clone())
                    confidences_history.append(confidence.clone())
                    quality_history.append(None if quality_scores is None else quality_scores.clone())
                    transfer_history.append(transfer_index.clone())
                    remask_history.append(remask_index.clone())
                    x0_histories.append(x0.clone())

        if not return_dict:
            return x
        return BaseSamplerOutput(
            sequences=x,
            histories=histories,
            confidences=confidences_history,
            quality_scores=quality_history,
            transfer_indices=transfer_history,
            remask_indices=remask_history,
            x0_histories=x0_histories,
        )

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config: MDLMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Fill in-place the mask tokens contained in `inputs`.

        Unlike sample(), this does NOT append new masks. Instead it finds
        existing mask tokens in the input and replaces them with predictions.
        Non-mask tokens are never modified.

        Args:
            inputs: List of token tensors/lists containing mask_token_id at positions to fill.
            config: Sampler configuration, or None for defaults.
            **kwargs: Override specific config parameters.
                revisitable_region: Optional editable-region mask [B, T] (bool).
                    In PRISM mode this region is eligible for remasking, even if
                    tokens are initially unmasked.

        Returns:
            BaseSamplerOutput (if return_dict=True) or raw tensor of token IDs.
        """
        if config is None:
            config = MDLMSamplerConfig()

        steps = kwargs.get("steps", config.steps)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get("stochastic_transfer", config.stochastic_transfer)
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get("begin_suppress_tokens", config.begin_suppress_tokens)
        prism_eta = kwargs.get("prism_eta", config.prism_eta)
        prism_quality_threshold = kwargs.get("prism_quality_threshold", config.prism_quality_threshold)
        prism_single_block_infill = kwargs.get("prism_single_block_infill", config.prism_single_block_infill)
        revisitable_region_override = kwargs.get("revisitable_region", None)

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # Handle empty prompts
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        if block_size is None:
            block_size = T

        if remasking == "prism" and prism_single_block_infill:
            # PRISM remasking can move masks across the whole editable region.
            # Using one block ensures every remasked token stays revisitable.
            block_size = T

        assert 1 <= block_size
        assert 1 <= steps

        # Build canvas: right-pad with EOS
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[i, :L] = 1

        # Track originally unmasked tokens for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        original_mask_region = (x == mask_id) & attention_mask.bool()
        if revisitable_region_override is None:
            revisitable_region = original_mask_region
        else:
            revisitable_region = torch.zeros((B, T), dtype=torch.bool, device=self.model.device)
            if isinstance(revisitable_region_override, torch.Tensor):
                rr = revisitable_region_override.to(device=self.model.device, dtype=torch.bool)
                if rr.dim() == 1:
                    if B != 1:
                        raise ValueError("1D revisitable_region is only valid for batch size 1")
                    width = min(rr.numel(), seq_lens[0], T)
                    revisitable_region[0, :width] = rr[:width]
                elif rr.dim() == 2:
                    if rr.size(0) != B:
                        raise ValueError("revisitable_region batch size mismatch")
                    for i in range(B):
                        width = min(rr.size(1), seq_lens[i], T)
                        revisitable_region[i, :width] = rr[i, :width]
                else:
                    raise ValueError("revisitable_region must be 1D or 2D")
            else:
                if len(revisitable_region_override) != B:
                    raise ValueError("revisitable_region list batch size mismatch")
                for i, mask_spec in enumerate(revisitable_region_override):
                    rr = torch.as_tensor(mask_spec, dtype=torch.bool, device=self.model.device)
                    width = min(rr.numel(), seq_lens[i], T)
                    revisitable_region[i, :width] = rr[:width]
            revisitable_region = revisitable_region & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # Block schedule over the entire sequence
        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None
        confidences_history = [] if return_dict else None
        quality_history = [] if return_dict else None
        transfer_history = [] if return_dict else None
        remask_history = [] if return_dict else None
        x0_histories = [] if return_dict else None

        for b in range(num_blocks):
            start = b * block_size
            stop = min(start + block_size, T)

            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=self.model.device
            )
            widths: list[int] = []
            for j in range(B):
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            if remasking == "prism":
                # PRISM must still run remasking steps even when the block starts
                # fully unmasked (warm-start correction).
                num_transfer_tokens = None
                effective_steps = steps_per_block
            else:
                num_transfer_tokens = get_num_transfer_tokens(
                    mask_index=block_mask_index,
                    steps=steps_per_block,
                    scheduler=self.scheduler,
                    stochastic=stochastic_transfer,
                )
                effective_steps = num_transfer_tokens.size(1)

            for s in range(effective_steps):
                is_final_step = (b == num_blocks - 1) and (s == effective_steps - 1)

                # Block-boundary clamping: confidence is -inf outside [start, start+width)
                # for each sequence.
                def _clamp(
                    x0_p: torch.Tensor, j: int,
                    _start: int = start, _widths: list = widths,
                ) -> None:
                    end_j = _start + _widths[j]
                    x0_p[j, :_start] = -np.inf
                    x0_p[j, end_j:] = -np.inf

                ctx = _StepContext(
                    x=x,
                    attention_mask=attention_mask,
                    unmasked_index=unmasked_index,
                    revisitable_region=revisitable_region,
                    block_clamp_fn=_clamp,
                    num_transfer_tokens=num_transfer_tokens,
                    step_idx=s,
                    total_steps=effective_steps,
                    is_final_step=is_final_step,
                    B=B,
                    mask_id=mask_id,
                )

                confidence, transfer_index, remask_index, x0, quality_scores = (
                    self._run_diffusion_step(
                        ctx=ctx,
                        cfg_scale=cfg_scale,
                        suppress_tokens=suppress_tokens,
                        begin_suppress_tokens=begin_suppress_tokens,
                        right_shift_logits=right_shift_logits,
                        temperature=temperature,
                        remasking=remasking,
                        stochastic_transfer=stochastic_transfer,
                        prism_eta=prism_eta,
                        prism_quality_threshold=prism_quality_threshold,
                    )
                )

                if histories is not None:
                    histories.append(x.clone())
                    confidences_history.append(confidence.clone())
                    quality_history.append(None if quality_scores is None else quality_scores.clone())
                    transfer_history.append(transfer_index.clone())
                    remask_history.append(remask_index.clone())
                    x0_histories.append(x0.clone())

        if not return_dict:
            return x
        return BaseSamplerOutput(
            sequences=x,
            histories=histories,
            confidences=confidences_history,
            quality_scores=quality_history,
            transfer_indices=transfer_history,
            remask_indices=remask_history,
            x0_histories=x0_histories,
        )


DiffusionSamplerConfig = MDLMSamplerConfig
DiffusionSampler = MDLMSampler
