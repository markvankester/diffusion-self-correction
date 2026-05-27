# Adapted from:
# https://github.com/ZHZisZZ/dllm

"""
Masked Diffusion Language Model (MDLM) Trainer.

References:

Simple and Effective Masked Diffusion Language Models:
https://arxiv.org/abs/2406.07524

Large Language Diffusion Models:
https://arxiv.org/abs/2502.09992
"""

from typing import Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from .schedules import BaseAlphaScheduler, LinearAlphaScheduler
from .interfaces import DiffusionModelLike, ModelOutputLike


@dataclass
class MDLMConfig(transformers.TrainingArguments):
    """Training arguments for the MDLM diffusion process."""
    time_epsilon: float = 1e-3
    loss_weight_type: str = "scheduler"  # "scheduler", "uniform"
    loss_norm_type: str = "token"        # "batch", "sequence", "token"
    right_shift_logits: bool = False


MDLMTrainerConfig = MDLMConfig  # Backwards-compatibility alias


class MDLMTrainer(transformers.Trainer):

    def __init__(
        self,
        args: MDLMConfig,
        scheduler: BaseAlphaScheduler | None = None,
        *pargs,
        **kwargs,
    ):
        super().__init__(args=args, *pargs, **kwargs)

        if not (0.0 < args.time_epsilon < 1.0):
            raise ValueError("time_epsilon must be in (0, 1)")

        self.scheduler = scheduler if scheduler is not None else LinearAlphaScheduler()
        self.time_epsilon = args.time_epsilon
        self.loss_weight_type = args.loss_weight_type
        self.loss_norm_type = args.loss_norm_type
        self.right_shift_logits = args.right_shift_logits       # False for bidirectional models

    def _preprocess_inputs(self, inputs):
        """
        Prepend BOS when right_shift_logits is enabled.

        Skipped if labels already start with -100 (idempotency guard for resumed steps).
        """
        if self.right_shift_logits:
            labels = inputs.get("labels", None)
            if labels is not None and torch.all(labels[:, 0] == -100):
                return inputs  # BOS already prepended
            inputs = _prepend_bos(
                inputs,
                bos_token_id=self.processing_class.bos_token_id,
                label_pad_token_id=-100,
            )
        return inputs

    def _postprocess_outputs(self, outputs: ModelOutputLike) -> ModelOutputLike:
        """Left-shift logits by one position for right-shift (causal) decoding mode."""
        if self.right_shift_logits:
            logits = outputs.logits
            outputs.logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return outputs

    def _compute_loss_weights(
        self,
        t: torch.Tensor,
        inputs: dict[str, Any],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Compute loss weights given timestep t and other arguments."""
        b, l = inputs["input_ids"].shape
        if self.loss_weight_type == "scheduler":
            loss_weights = self.scheduler.weight(t).unsqueeze(1).repeat(1, l)
        elif self.loss_weight_type == "uniform":
            loss_weights = torch.ones_like(inputs["input_ids"])
        else:
            raise NotImplementedError
        return loss_weights

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().contiguous()

        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().contiguous()

        return (loss.detach(), logits, labels)

    def _sample_diffusion_mask(self, input_ids: torch.Tensor, maskable_mask: torch.Tensor):
        """
        Sample diffusion timesteps and apply stochastic masking.
        Returns:
            t: The sampled timesteps.
            masked_mask: Boolean mask indicating which tokens were stochastically masked.
            noised_input_ids: The input_ids with masked tokens replaced by the mask token.
        """
        b, l = input_ids.shape
        # === 1. Sample diffusion timesteps ===
        # t ∈ [ε, 1); p_mask = 1 - α(t) is the per-token masking probability.
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )  # [b]
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)  # [b, l]

        # === 2. Apply stochastic masking ===
        # Mask each valid position independently; positions with label=-100 are excluded.
        masked_mask = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )
        return t, masked_mask, noised_input_ids

    def _valid_training_mask(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Positions eligible for diffusion/correction losses."""
        valid_mask = labels != -100
        pad_id = getattr(self.processing_class, "pad_token_id", None)
        if pad_id is not None:
            valid_mask = valid_mask & (input_ids != pad_id)
        if attention_mask is not None:
            valid_mask = valid_mask & attention_mask.bool()
        return valid_mask

    def _suppress_artifact_special_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        """Prevent correction-artifact sampling from producing impossible tokens."""
        suppressed_ids = []
        for name in ("pad_token_id", "mask_token_id"):
            token_id = getattr(self.processing_class, name, None)
            if token_id is not None:
                suppressed_ids.append(int(token_id))
        if not suppressed_ids:
            return logits
        logits = logits.clone()
        ids = torch.as_tensor(sorted(set(suppressed_ids)), device=logits.device, dtype=torch.long)
        ids = ids[(ids >= 0) & (ids < logits.size(-1))]
        if ids.numel() > 0:
            logits[..., ids] = -torch.inf
        return logits

    def compute_loss(
        self,
        model: DiffusionModelLike | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute the masked diffusion language modeling loss.

        Applies stochastic masking to input tokens based on a diffusion timestep,
        then computes the weighted cross-entropy loss for predicting the original tokens.

        Args:
            model: The language model to train.
            inputs: Dictionary containing input_ids, labels, and optionally attention_mask.
            return_outputs: If True, return both loss and model outputs.

        Returns:
            Loss tensor, or tuple of (loss, outputs) if return_outputs is True.
        """
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape
        maskable_mask = labels != -100  # [b, l]

        t, masked_mask, noised_input_ids = self._sample_diffusion_mask(input_ids, maskable_mask)

        # === 3. Forward pass ===
        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        # === 4. Compute per-token loss weights ===
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_mask=masked_mask
        )

        # === 5. Compute weighted cross-entropy ===
        assert (
            input_ids[maskable_mask] == labels[maskable_mask]
        ).all(), "Mismatch between input_ids and labels at valid positions"

        token_nll_raw = F.cross_entropy(
            logits.transpose(1, 2),  # [b, V, l]
            input_ids,               # [b, l]
            reduction="none",
        )
        token_nll = token_nll_raw * loss_weights * masked_mask.to(token_nll_raw.dtype)  # [b, l]

        # === 6. Normalize loss ===
        if self.loss_norm_type == "token":
            token_nll /= maskable_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_nll /= maskable_mask.sum(-1, keepdim=True).clamp_min(1) * b
        elif self.loss_norm_type == "batch":
            token_nll /= b
        else:
            raise ValueError("Invalid loss_norm_type.")
        loss = token_nll.sum()

        # === 7. Log MDLM-specific diagnostics to wandb ===
        if self.model.training and _WANDB_AVAILABLE and wandb.run is not None:
            n_masked = masked_mask.sum().item()

            with torch.no_grad():
                raw_nll = (token_nll_raw.detach() * masked_mask.to(token_nll_raw.dtype)).sum() / max(n_masked, 1)
                correct = (logits.detach().argmax(dim=-1) == input_ids) & masked_mask
                accuracy = correct.sum().item() / max(n_masked, 1)
            wandb.log(
                {
                    "train/unweighted_nll": raw_nll.item(),
                    "train/masked_token_accuracy": accuracy,
                },
                step=self.state.global_step,
            )

        return (loss, outputs) if return_outputs else loss


def _prepend_bos(
    inputs: dict[str, torch.Tensor],
    bos_token_id: int,
    label_pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Prepends the BOS token to the input sequences."""
    input_ids = inputs["input_ids"]
    labels = inputs.get("labels", None)
    attention_mask = inputs.get("attention_mask", None)

    b = input_ids.size(0)
    bos_col = torch.full((b, 1), bos_token_id, dtype=input_ids.dtype, device=input_ids.device)
    inputs["input_ids"] = torch.cat([bos_col, input_ids], dim=1)

    if labels is not None:
        ignore_col = torch.full((b, 1), label_pad_token_id, dtype=labels.dtype, device=labels.device)
        inputs["labels"] = torch.cat([ignore_col, labels], dim=1)

    if attention_mask is not None:
        att_col = torch.ones((b, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        inputs["attention_mask"] = torch.cat([att_col, attention_mask], dim=1)

    return inputs


# Backwards-compatibility aliases
DiffusionTrainer = MDLMTrainer
DiffusionTrainingConfig = MDLMConfig
