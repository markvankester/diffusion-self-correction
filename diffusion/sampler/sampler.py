from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import DiffusionModelLike, TokenizerLike
from ..remdm import compute_sigma, confidence_reweight
from ..schedules import BaseAlphaScheduler, LinearAlphaScheduler
from ..utils import add_gumbel_noise, get_num_transfer_tokens

from .config import (
    BaseSamplerOutput,
    BaseSamplerConfig,
    MDLMSamplerConfig,
    unpack_sampler_config,
)


@dataclass
class _StepContext:
    """
    Carries the per-block/per-step context that differs between sample() and infill().
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
    confidence_scores: torch.Tensor | None = None
    blocked_remask_indices: list[list[int]] | None = None
    t_tensor: torch.Tensor | None = None
    delta_t: float | None = None
    prompt_lens: list[int] | None = None
    block_idx: int | None = None
    block_size: int | None = None


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
        backplay_head: Optional BackPlayHead for error-guided remasking.
    """
    model: DiffusionModelLike
    tokenizer: TokenizerLike
    scheduler: BaseAlphaScheduler | None = None
    prism_head: nn.Module | None = None
    backplay_head: nn.Module | None = None

    def __post_init__(self):
        if self.scheduler is None:
            self.scheduler = LinearAlphaScheduler()

    # ------------------------------------------------------------------
    # Remasking strategies helpers
    # ------------------------------------------------------------------

    def _step_prism(
        self,
        ctx: _StepContext,
        x0: torch.Tensor,
        confidence: torch.Tensor,
        mask_index: torch.Tensor,
        stochastic_transfer: bool,
        prism_eta: float,
        prism_quality_threshold: float | None,
        hidden_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = ctx.x
        attention_mask = ctx.attention_mask
        mask_id = ctx.mask_id
        B = ctx.B
        remask_index = torch.zeros_like(x, dtype=torch.bool)
        quality_scores = None

        if not ctx.is_final_step and self.prism_head is not None and prism_eta > 0.0:
            if hidden_states is None:
                raise RuntimeError("PRISM remasking requires hidden states from the backbone")
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

        transfer_index = torch.zeros_like(x, dtype=torch.bool)
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

            extra_unmask = int(remask_index[j].sum().item())
            k_commit = min(n_masked, base_unmask + extra_unmask)
            if k_commit > 0:
                _, select_idx = torch.topk(confidence[j], k=k_commit)
                transfer_index[j, select_idx] = True

        x[transfer_index] = x0[transfer_index]
        x[remask_index] = mask_id

        return transfer_index, remask_index, quality_scores

    def _step_backplay(
        self,
        ctx: _StepContext,
        x0: torch.Tensor,
        confidence: torch.Tensor,
        mask_index: torch.Tensor,
        stochastic_transfer: bool,
        backplay_budget: int,
        backplay_threshold: float,
        backplay_stride: int,
        backplay_block_buffer: int,
        hidden_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = ctx.x
        attention_mask = ctx.attention_mask
        mask_id = ctx.mask_id
        B = ctx.B
        remask_index = torch.zeros_like(x, dtype=torch.bool)
        quality_scores = None

        if self.backplay_head is None:
            raise RuntimeError("BackPlay remasking requires a backplay_head")
        if hidden_states is None:
            raise RuntimeError("BackPlay remasking requires hidden states from the backbone")

        if ctx.t_tensor is not None and ctx.delta_t is not None:
            t_curr = ctx.t_tensor
            delta_t = ctx.delta_t
        else:
            t_curr = torch.full((B,), (ctx.total_steps - ctx.step_idx) / ctx.total_steps, device=x.device)
            delta_t = 1.0 / ctx.total_steps

        if not ctx.is_final_step and ctx.step_idx % max(backplay_stride, 1) == 0:
            error_scores = self.backplay_head(hidden_states, attention_mask=attention_mask)
            quality_scores = 1.0 - error_scores
            is_clean = (x != mask_id) & attention_mask.bool() & ctx.revisitable_region
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is not None:
                is_clean = is_clean & (x != pad_id)
            for j in range(B):
                if t_curr[j] <= 0:
                    continue
                eligible = is_clean[j].clone()
                if ctx.blocked_remask_indices is not None:
                    blocked = ctx.blocked_remask_indices[j]
                    if blocked:
                        blocked_idx = torch.as_tensor(blocked, device=x.device, dtype=torch.long)
                        blocked_idx = blocked_idx[(blocked_idx >= 0) & (blocked_idx < x.size(1))]
                        eligible[blocked_idx] = False
                eligible = eligible & (error_scores[j] > backplay_threshold)
                n_eligible = int(eligible.sum().item())
                if n_eligible == 0:
                    continue
                k_remask = min(max(backplay_budget, 0), n_eligible)
                if k_remask <= 0:
                    continue
                row_scores = error_scores[j].clone()
                row_scores[~eligible] = -torch.inf
                _, to_remask_idx = torch.topk(row_scores, k=k_remask)
                remask_index[j, to_remask_idx] = True

        current_mask_region = mask_index & ctx.revisitable_region
        if ctx.t_tensor is None or ctx.delta_t is None:
            step_t = (ctx.total_steps - ctx.step_idx) / ctx.total_steps
            prev_t = (ctx.total_steps - ctx.step_idx - 1) / ctx.total_steps
            reverse_transfer_prob = 1 - self.scheduler.reverse_mask_prob(s=prev_t, t=step_t)
        else:
            reverse_transfer_prob = None

        transfer_index = torch.zeros_like(x, dtype=torch.bool)
        for j in range(B):
            if t_curr[j] <= 0:
                continue
            n_masked = int(current_mask_region[j].sum().item())
            if n_masked == 0:
                continue

            if ctx.t_tensor is not None and ctx.delta_t is not None:
                tnew_j = max(float(t_curr[j].item()) - delta_t, 0.0)
                L_j = int(ctx.revisitable_region[j].sum().item())
                base_unmask = int(round(L_j * float(t_curr[j].item()))) - int(round(L_j * tnew_j))
                base_unmask = max(0, base_unmask)
            else:
                if stochastic_transfer:
                    base_unmask = int(
                        torch.distributions.Binomial(
                            torch.tensor(float(n_masked), device=x.device),
                            torch.tensor(float(reverse_transfer_prob), device=x.device),
                        ).sample().item()
                    )
                else:
                    base_unmask = int(round(n_masked * float(reverse_transfer_prob)))

            extra_unmask = 0 if ctx.t_tensor is not None else int(remask_index[j].sum().item())
            k_commit = min(n_masked, base_unmask + extra_unmask)
            if k_commit > 0:
                _, select_idx = torch.topk(confidence[j], k=k_commit)
                transfer_index[j, select_idx] = True

        x[transfer_index] = x0[transfer_index]
        x[remask_index] = mask_id

        if ctx.blocked_remask_indices is not None and backplay_block_buffer > 0:
            for j in range(B):
                new_indices = remask_index[j].nonzero(as_tuple=True)[0].tolist()
                if not new_indices:
                    continue
                ctx.blocked_remask_indices[j].extend(int(idx) for idx in new_indices)
                overflow = len(ctx.blocked_remask_indices[j]) - backplay_block_buffer
                if overflow > 0:
                    del ctx.blocked_remask_indices[j][:overflow]

        if ctx.t_tensor is not None and ctx.delta_t is not None:
            for j in range(B):
                if t_curr[j] <= 0:
                    continue
                tnew_j = max(float(t_curr[j].item()) - delta_t, 0.0)
                L_j = int(ctx.revisitable_region[j].sum().item())
                extra_unmask = int(remask_index[j].sum().item())
                t_curr[j] = tnew_j + (extra_unmask / max(1, L_j))

        return transfer_index, remask_index, quality_scores

    def _step_remdm_conf(
        self,
        ctx: _StepContext,
        x0: torch.Tensor,
        x0_p: torch.Tensor,
        confidence: torch.Tensor,
        mask_index: torch.Tensor,
        stochastic_transfer: bool,
        remdm_eta_rescale: float,
        remdm_eta_cap: float,
        remdm_ton: float,
        remdm_toff: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = ctx.x
        mask_id = ctx.mask_id
        B = ctx.B
        remask_index = torch.zeros_like(x, dtype=torch.bool)
        quality_scores = None

        assert ctx.confidence_scores is not None, (
            "remdm_conf requires confidence_scores in _StepContext"
        )

        step_t = (ctx.total_steps - ctx.step_idx) / ctx.total_steps
        prev_t = (ctx.total_steps - ctx.step_idx - 1) / ctx.total_steps
        alpha_t = float(self.scheduler.alpha(step_t))
        alpha_s = float(self.scheduler.alpha(prev_t))
        sigma = compute_sigma(alpha_s, alpha_t, remdm_eta_rescale, remdm_eta_cap)

        if step_t > remdm_ton or step_t <= remdm_toff:
            sigma = 0.0

        if not ctx.is_final_step and sigma > 0:
            sigma_per_token = confidence_reweight(
                sigma_base=sigma,
                psi_scores=ctx.confidence_scores,
                mask_index=mask_index,
                revisitable_region=ctx.revisitable_region,
            )
            remask_probs = torch.rand_like(sigma_per_token)
            remask_index = (remask_probs < sigma_per_token)
            remask_index = remask_index & (~mask_index) & ctx.revisitable_region

        current_mask_region = mask_index & ctx.revisitable_region
        reverse_transfer_prob = 1 - self.scheduler.reverse_mask_prob(s=prev_t, t=step_t)

        transfer_index = torch.zeros_like(x, dtype=torch.bool)
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

            extra_unmask = int(remask_index[j].sum().item())
            k_commit = min(n_masked, base_unmask + extra_unmask)
            if k_commit > 0:
                _, select_idx = torch.topk(confidence[j], k=k_commit)
                transfer_index[j, select_idx] = True

        x[transfer_index] = x0[transfer_index]
        x[remask_index] = mask_id

        ctx.confidence_scores[transfer_index] = x0_p[transfer_index]
        ctx.confidence_scores[remask_index] = float("inf")

        return transfer_index, remask_index, quality_scores

    def _step_standard(
        self,
        ctx: _StepContext,
        x0: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = ctx.x
        B = ctx.B
        remask_index = torch.zeros_like(x, dtype=torch.bool)
        quality_scores = None

        assert ctx.num_transfer_tokens is not None

        transfer_index = torch.zeros_like(x, dtype=torch.bool)
        for j in range(B):
            k = int(ctx.num_transfer_tokens[j, ctx.step_idx].item())
            if k > 0:
                _, select_idx = torch.topk(confidence[j], k=k)
                transfer_index[j, select_idx] = True
        x[transfer_index] = x0[transfer_index]

        return transfer_index, remask_index, quality_scores

    def _step_remedi(
        self,
        ctx: _StepContext,
        x0: torch.Tensor,
        x0_p: torch.Tensor,
        prompt_lens: list[int] | None,
        block_idx: int | None,
        block_size: int | None,
        model_confidences: torch.Tensor | None = None,
        remedi_threshold: float = float("inf"),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = ctx.x
        mask_id = ctx.mask_id
        B = ctx.B
        quality_scores = None

        assert ctx.num_transfer_tokens is not None
        if block_idx is None or block_size is None:
            raise ValueError("block_idx and block_size must be specified in _StepContext for RemeDi remasking")
        if prompt_lens is None:
            prompt_lens = [0] * B

        transfer_index = torch.zeros_like(x, dtype=torch.bool)
        remask_index = torch.zeros_like(x, dtype=torch.bool)

        scores_source = model_confidences if model_confidences is not None else x0_p

        for j in range(B):
            active_mask = torch.zeros_like(x[j], dtype=torch.bool)
            start = prompt_lens[j] + block_idx * block_size
            end = min(start + block_size, x.size(1))
            if start < end:
                active_mask[start:end] = ctx.revisitable_region[j, start:end]

            if remedi_threshold < float("inf"):
                remask_index[j, active_mask] = (scores_source[j, active_mask] < remedi_threshold)
            else:
                remask_index[j, active_mask] = True

            k = int(ctx.num_transfer_tokens[j, : ctx.step_idx + 1].sum().item())
            n_active = int(active_mask.sum().item())
            k = min(k, n_active)

            if k > 0:
                scores = scores_source[j].clone()
                scores[~active_mask] = -torch.inf
                _, select_idx = torch.topk(scores, k=k)
                
                transfer_index[j, select_idx] = True
                remask_index[j, select_idx] = False

        x[transfer_index] = x0[transfer_index]
        x[remask_index] = mask_id

        return transfer_index, remask_index, quality_scores

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
        backplay_budget: int = 2,
        backplay_threshold: float = 0.75,
        backplay_stride: int = 4,
        backplay_block_buffer: int = 4,
        remdm_eta_rescale: float = 1.0,
        remdm_eta_cap: float = 1.0,
        remdm_ton: float = 1.0,
        remdm_toff: float = 0.0,
        remedi_threshold: float = float("inf"),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = ctx.x
        attention_mask = ctx.attention_mask
        mask_id = ctx.mask_id
        B = ctx.B

        mask_index = (x == mask_id)
        if remasking == "backplay" and self.backplay_head is None:
            raise RuntimeError("BackPlay remasking requires a backplay_head")

        # ---- 1. Forward pass ------------------------------------------------
        model_conf = None
        if cfg_scale > 0.0:
            un_x = x.clone()
            un_x[ctx.unmasked_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            outputs = self.model(
                x_, attention_mask=attention_mask.repeat(2, 1)
            )
            logits = outputs.logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            
            model_conf_all = getattr(outputs, "confidences", None)
            if model_conf_all is not None:
                model_conf, _ = torch.chunk(model_conf_all, 2, dim=0)

            needs_hidden_states = (
                (remasking == "prism" and self.prism_head is not None and prism_eta > 0.0)
                or (remasking == "backplay" and self.backplay_head is not None)
            )
            if needs_hidden_states:
                correction_outputs = self.model(
                    input_ids=x,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                if remasking == "backplay":
                    hidden_index = getattr(self.backplay_head.config, "hidden_state_index", -2)
                    hidden_states = correction_outputs.hidden_states[hidden_index]
                else:
                    hidden_states = correction_outputs.hidden_states[-1]
            else:
                hidden_states = None
        else:
            output_hidden_states = remasking in ("prism", "backplay")
            outputs = self.model(
                input_ids=x,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
            )
            logits = outputs.logits
            model_conf = getattr(outputs, "confidences", None)
            if output_hidden_states and remasking == "backplay":
                hidden_index = getattr(self.backplay_head.config, "hidden_state_index", -2)
                hidden_states = outputs.hidden_states[hidden_index]
            elif output_hidden_states:
                hidden_states = outputs.hidden_states[-1]
            else:
                hidden_states = None

        # ---- 2. Token suppression / logit shift ----------------------------
        if suppress_tokens is not None and len(suppress_tokens) > 0:
            suppress_ids = torch.as_tensor(suppress_tokens, dtype=torch.long, device=logits.device)
            logits[:, :, suppress_ids] = -torch.inf

        if right_shift_logits:
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
            begin_suppress_ids = torch.as_tensor(begin_suppress_tokens, dtype=torch.long, device=logits.device)
            logits[:, :, begin_suppress_ids] = -torch.inf

        # ---- 3. Candidate tokens -------------------------------------------
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        # ---- 4. Per-position confidence ------------------------------------
        if remasking in ("low_confidence", "prism", "backplay", "remdm_conf", "remedi"):
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        elif remasking == "random":
            x0_p = torch.rand_like(x0, dtype=torch.float32)
        else:
            raise NotImplementedError(f"Unknown remasking strategy: {remasking!r}")

        for j in range(B):
            ctx.block_clamp_fn(x0_p, j)

        x0 = torch.where(mask_index, x0, x)
        if remasking == "remedi":
            confidence = model_conf if model_conf is not None else x0_p
        else:
            confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x.device))

        # ---- 5. Select commit and remask -----------------------------------
        if remasking == "prism":
            transfer_index, remask_index, quality_scores = self._step_prism(
                ctx=ctx,
                x0=x0,
                confidence=confidence,
                mask_index=mask_index,
                stochastic_transfer=stochastic_transfer,
                prism_eta=prism_eta,
                prism_quality_threshold=prism_quality_threshold,
                hidden_states=hidden_states,
            )
        elif remasking == "backplay":
            transfer_index, remask_index, quality_scores = self._step_backplay(
                ctx=ctx,
                x0=x0,
                confidence=confidence,
                mask_index=mask_index,
                stochastic_transfer=stochastic_transfer,
                backplay_budget=backplay_budget,
                backplay_threshold=backplay_threshold,
                backplay_stride=backplay_stride,
                backplay_block_buffer=backplay_block_buffer,
                hidden_states=hidden_states,
            )
        elif remasking == "remdm_conf":
            transfer_index, remask_index, quality_scores = self._step_remdm_conf(
                ctx=ctx,
                x0=x0,
                x0_p=x0_p,
                confidence=confidence,
                mask_index=mask_index,
                stochastic_transfer=stochastic_transfer,
                remdm_eta_rescale=remdm_eta_rescale,
                remdm_eta_cap=remdm_eta_cap,
                remdm_ton=remdm_ton,
                remdm_toff=remdm_toff,
            )
        elif remasking == "remedi":
            transfer_index, remask_index, quality_scores = self._step_remedi(
                ctx=ctx,
                x0=x0,
                x0_p=x0_p,
                prompt_lens=ctx.prompt_lens,
                block_idx=ctx.block_idx,
                block_size=ctx.block_size,
                model_confidences=model_conf,
                remedi_threshold=remedi_threshold,
            )
        else:
            transfer_index, remask_index, quality_scores = self._step_standard(
                ctx=ctx,
                x0=x0,
                confidence=confidence,
            )

        return confidence, transfer_index, remask_index, x0, quality_scores

    # ------------------------------------------------------------------
    # Unifed Loop Execution
    # ------------------------------------------------------------------

    def _run_block_loop(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        unmasked_index: torch.Tensor,
        revisitable_region: torch.Tensor,
        num_blocks: int,
        block_size: int,
        steps_per_block: int,
        remasking: str,
        stochastic_transfer: bool,
        confidence_scores: torch.Tensor | None,
        get_block_mask_and_clamp_fn: callable,
        params: dict,
    ) -> tuple:
        B = x.size(0)
        mask_id = self.tokenizer.mask_token_id
        return_dict = params["return_dict"]

        histories = [x.clone()] if return_dict else None
        confidences_history = [] if return_dict else None
        quality_history = [] if return_dict else None
        transfer_history = [] if return_dict else None
        remask_history = [] if return_dict else None
        x0_histories = [] if return_dict else None

        blocked_remask_indices = [[] for _ in range(B)] if remasking == "backplay" else None

        for b in range(num_blocks):
            block_mask_index, block_clamp_fn = get_block_mask_and_clamp_fn(b)

            if remasking in ("prism", "backplay", "remdm_conf"):
                effective_steps = steps_per_block
                num_transfer_tokens = None
            else:
                num_transfer_tokens = get_num_transfer_tokens(
                    mask_index=block_mask_index,
                    steps=steps_per_block,
                    scheduler=self.scheduler,
                    stochastic=stochastic_transfer,
                )
                effective_steps = num_transfer_tokens.size(1)

            if remasking == "backplay":
                # Run dynamic while loop for BackPlay (Algorithm 1)
                delta_t = 1.0 / steps_per_block
                t_tensor = torch.ones(B, device=x.device, dtype=torch.float32)
                s = 0
                max_allowed_steps = steps_per_block * 3

                while (t_tensor > 0.0).any() and s < max_allowed_steps:
                    is_final_step = (s == max_allowed_steps - 1) or not (t_tensor > delta_t).any()

                    ctx = _StepContext(
                        x=x,
                        attention_mask=attention_mask,
                        unmasked_index=unmasked_index,
                        revisitable_region=revisitable_region,
                        block_clamp_fn=block_clamp_fn,
                        num_transfer_tokens=None,
                        step_idx=s,
                        total_steps=steps_per_block,
                        is_final_step=is_final_step,
                        B=B,
                        mask_id=mask_id,
                        confidence_scores=None,
                        blocked_remask_indices=blocked_remask_indices,
                        t_tensor=t_tensor,
                        delta_t=delta_t,
                    )

                    confidence, transfer_index, remask_index, x0, quality_scores = (
                        self._run_diffusion_step(
                            ctx=ctx,
                            cfg_scale=params["cfg_scale"],
                            suppress_tokens=params["suppress_tokens"],
                            begin_suppress_tokens=params["begin_suppress_tokens"],
                            right_shift_logits=params["right_shift_logits"],
                            temperature=params["temperature"],
                            remasking=remasking,
                            stochastic_transfer=stochastic_transfer,
                            prism_eta=params["prism_eta"],
                            prism_quality_threshold=params["prism_quality_threshold"],
                            backplay_budget=params["backplay_budget"],
                            backplay_threshold=params["backplay_threshold"],
                            backplay_stride=params["backplay_stride"],
                            backplay_block_buffer=params["backplay_block_buffer"],
                            remdm_eta_rescale=params["remdm_eta_rescale"],
                            remdm_eta_cap=params["remdm_eta_cap"],
                            remdm_ton=params["remdm_ton"],
                            remdm_toff=params["remdm_toff"],
                        )
                    )

                    if histories is not None:
                        histories.append(x.clone())
                        confidences_history.append(confidence.clone())
                        quality_history.append(None if quality_scores is None else quality_scores.clone())
                        transfer_history.append(transfer_index.clone())
                        remask_history.append(remask_index.clone())
                        x0_histories.append(x0.clone())

                    s += 1
            else:
                for s in range(effective_steps):
                    is_final_step = (b == num_blocks - 1) and (s == effective_steps - 1)

                    ctx = _StepContext(
                        x=x,
                        attention_mask=attention_mask,
                        unmasked_index=unmasked_index,
                        revisitable_region=revisitable_region,
                        block_clamp_fn=block_clamp_fn,
                        num_transfer_tokens=num_transfer_tokens,
                        step_idx=s,
                        total_steps=effective_steps,
                        is_final_step=is_final_step,
                        B=B,
                        mask_id=mask_id,
                        confidence_scores=confidence_scores,
                        blocked_remask_indices=blocked_remask_indices,
                        prompt_lens=params.get("prompt_lens"),
                        block_idx=b,
                        block_size=block_size,
                    )

                    confidence, transfer_index, remask_index, x0, quality_scores = (
                        self._run_diffusion_step(
                            ctx=ctx,
                            cfg_scale=params["cfg_scale"],
                            suppress_tokens=params["suppress_tokens"],
                            begin_suppress_tokens=params["begin_suppress_tokens"],
                            right_shift_logits=params["right_shift_logits"],
                            temperature=params["temperature"],
                            remasking=remasking,
                            stochastic_transfer=stochastic_transfer,
                            prism_eta=params["prism_eta"],
                            prism_quality_threshold=params["prism_quality_threshold"],
                            backplay_budget=params["backplay_budget"],
                            backplay_threshold=params["backplay_threshold"],
                            backplay_stride=params["backplay_stride"],
                            backplay_block_buffer=params["backplay_block_buffer"],
                            remdm_eta_rescale=params["remdm_eta_rescale"],
                            remdm_eta_cap=params["remdm_eta_cap"],
                            remdm_ton=params["remdm_ton"],
                            remdm_toff=params["remdm_toff"],
                            remedi_threshold=params.get("remedi_threshold", float("inf")),
                        )
                    )

                    if histories is not None:
                        histories.append(x.clone())
                        if remasking == "remdm_conf" and confidence_scores is not None:
                            confidences_history.append(confidence_scores.clone())
                        else:
                            confidences_history.append(confidence.clone())
                        quality_history.append(None if quality_scores is None else quality_scores.clone())
                        transfer_history.append(transfer_index.clone())
                        remask_history.append(remask_index.clone())
                        x0_histories.append(x0.clone())

        return histories, confidences_history, quality_history, transfer_history, remask_history, x0_histories

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

        params = unpack_sampler_config(config, kwargs)
        block_size = params["block_size"]
        steps = params["steps"]
        max_new_tokens = params["max_new_tokens"]
        max_length = params["max_length"]
        right_shift_logits = params["right_shift_logits"]
        remasking = params["remasking"]
        stochastic_transfer = params["stochastic_transfer"]
        cfg_keep_tokens = params["cfg_keep_tokens"]
        return_dict = params["return_dict"]

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

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
        params["prompt_lens"] = prompt_lens

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        if remasking in ("prism", "backplay", "remdm_conf"):
            block_size = max_new_tokens

        B = len(inputs)
        T = max_length

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = mask_id
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        gen_region_mask = torch.zeros((B, T), dtype=torch.bool, device=x.device)
        for j in range(B):
            gen_region_mask[j, prompt_lens[j] : prompt_lens[j] + max_new_tokens] = True

        num_blocks = math.ceil(max_new_tokens / block_size)
        base_steps = math.ceil(steps / num_blocks)

        confidence_scores = (
            torch.full((B, T), float("inf"), device=x.device)
            if remasking == "remdm_conf" else None
        )

        def get_block_mask_and_clamp_fn(b: int):
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )
            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    if remasking == "remedi":
                        block_mask_index[j, :width] = gen_region_mask[j, start:end]
                    else:
                        block_mask_index[j, :width] = (x[j, start:end] == mask_id)

            def _clamp(x0_p: torch.Tensor, j: int, _b: int = b, _pl: list = prompt_lens) -> None:
                x0_p[j, _pl[j] + (_b + 1) * block_size :] = -np.inf

            return block_mask_index, _clamp

        histories, confidences_history, quality_history, transfer_history, remask_history, x0_histories = (
            self._run_block_loop(
                x=x,
                attention_mask=attention_mask,
                unmasked_index=unmasked_index,
                revisitable_region=gen_region_mask,
                num_blocks=num_blocks,
                block_size=block_size,
                steps_per_block=base_steps,
                remasking=remasking,
                stochastic_transfer=stochastic_transfer,
                confidence_scores=confidence_scores,
                get_block_mask_and_clamp_fn=get_block_mask_and_clamp_fn,
                params=params,
            )
        )

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
                    In PRISM/ReMDM mode this region is eligible for remasking, even if
                    tokens are initially unmasked.
                initial_confidence: Optional [B, T] tensor of pre-computed ψ scores
                    for ReMDM-conf. Use compute_initial_confidence() to get fair
                    scores for tokens that were never predicted by the model.

        Returns:
            BaseSamplerOutput (if return_dict=True) or raw tensor of token IDs.
        """
        if config is None:
            config = MDLMSamplerConfig()

        params = unpack_sampler_config(config, kwargs)
        block_size = params["block_size"]
        steps = params["steps"]
        right_shift_logits = params["right_shift_logits"]
        remasking = params["remasking"]
        stochastic_transfer = params["stochastic_transfer"]
        cfg_keep_tokens = params["cfg_keep_tokens"]
        return_dict = params["return_dict"]
        prism_single_block_infill = params["prism_single_block_infill"]
        
        revisitable_region_override = kwargs.get("revisitable_region", None)
        initial_confidence = kwargs.get("initial_confidence", None)

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

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
        params["prompt_lens"] = [0] * B
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        if block_size is None:
            block_size = T

        if remasking in ("prism", "backplay") and prism_single_block_infill:
            block_size = T

        if remasking == "remdm_conf":
            block_size = T

        assert 1 <= block_size
        assert 1 <= steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[i, :L] = 1

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

        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(steps / num_blocks)

        if remasking == "remdm_conf":
            if initial_confidence is not None:
                confidence_scores = initial_confidence.to(
                    device=self.model.device, dtype=torch.float32
                ).clone()
            else:
                confidence_scores = torch.full(
                    (B, T), float("inf"), device=self.model.device
                )
            confidence_scores[x == mask_id] = float("inf")
            confidence_scores[~revisitable_region] = float("inf")
        else:
            confidence_scores = None

        def get_block_mask_and_clamp_fn(b: int):
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
                    if remasking == "remedi":
                        block_mask_index[j, :width] = revisitable_region[j, start : start + width]
                    else:
                        block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            def _clamp(
                x0_p: torch.Tensor, j: int,
                _start: int = start, _widths: list = widths,
            ) -> None:
                end_j = _start + _widths[j]
                x0_p[j, :_start] = -np.inf
                x0_p[j, end_j:] = -np.inf

            return block_mask_index, _clamp

        histories, confidences_history, quality_history, transfer_history, remask_history, x0_histories = (
            self._run_block_loop(
                x=x,
                attention_mask=attention_mask,
                unmasked_index=unmasked_index,
                revisitable_region=revisitable_region,
                num_blocks=num_blocks,
                block_size=block_size,
                steps_per_block=steps_per_block,
                remasking=remasking,
                stochastic_transfer=stochastic_transfer,
                confidence_scores=confidence_scores,
                get_block_mask_and_clamp_fn=get_block_mask_and_clamp_fn,
                params=params,
            )
        )

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


DiffusionSampler = MDLMSampler
