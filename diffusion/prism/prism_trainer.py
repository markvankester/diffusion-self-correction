"""
PRISM Trainer
=============
Fine-tuning trainer implementing Algorithm 1 from the PRISM paper.

Extends MDLMTrainer to add:
  - PRISM loss: BCE(1[x_i == y_i], g_θ^i(y)) for per-token quality learning
  - MDM regularization loss to preserve the backbone's unmasking ability
  - Stop-gradient on f_θ when sampling y from z
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import json
from typing import Any
from dataclasses import dataclass
from pathlib import Path

from ..interfaces import DiffusionModelLike
from ..trainer import MDLMConfig, MDLMTrainer
from ..schedules import BaseAlphaScheduler, LinearAlphaScheduler
from .prism_head import PRISMHead


@dataclass
class PRISMConfig(MDLMConfig):
    """Configuration for PRISM fine-tuning, extending the MDM trainer config."""
    prism_lambda: float = 5.0       # Regularization weight for MDM loss
    prism_k: int = 5                # Number of masked tokens to unmask per sample
    prism_freeze_unmasking_head: bool = True

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()


class PRISMTrainer(MDLMTrainer):
    """
    PRISM fine-tuning trainer (Algorithm 1 from the paper).

    Overrides compute_loss to jointly optimize:
      1. PRISM loss: BCE on per-token quality predictions
      2. MDM regularization: standard masked diffusion loss to preserve f_θ

    Args:
        prism_head: The PRISMHead module for per-token quality scoring.
        All other args are passed to the parent MDLMTrainer.
    """

    def __init__(
        self,
        args: PRISMConfig,
        prism_head: PRISMHead = None,
        *pargs,
        **kwargs,
    ):
        super().__init__(args=args, *pargs, **kwargs)
        if prism_head is None:
            raise ValueError("prism_head must be provided to PRISMTrainer")
        self.prism_head = prism_head
        # Guard against duplicate metric logs when gradient accumulation is enabled.
        # compute_loss() runs per micro-batch, but we only want one PRISM log per optimizer step.
        self._last_prism_log_step = -1

    def compute_loss(
        self,
        model: DiffusionModelLike | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute the PRISM fine-tuning loss (Algorithm 1).

        Steps:
          (a) Sample (x, z): mask data tokens to get noised input z
          (b) Forward pass on z -> logits for MDM loss and sampling y
          (c) Sample y: unmask k positions using f_sg(theta) (stop-gradient)
          (d) Forward pass on y -> hidden states for the quality head
          (e) PRISM BCE loss + MDM regularization loss
        """
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape
        maskable_mask = labels != -100

        # (a) Sample (x, z)
        t, masked_mask, z = self._sample_diffusion_mask(input_ids, maskable_mask)

        # (b) Forward pass on z
        outputs_z = model(input_ids=z, attention_mask=attention_mask, output_hidden_states=True)
        outputs_z = self._postprocess_outputs(outputs_z)
        logits_z = outputs_z.logits  # [b, l, V]

        # MDM regularization: (1/n) sum_{j: z_j=m} -log f_theta(x_j | z), averaged over batch
        # n = sequence length, matching Eq. 3 / Algorithm 1 line 10.
        token_nll = F.cross_entropy(logits_z.transpose(1, 2), input_ids, reduction="none")
        token_nll = token_nll * masked_mask.to(token_nll.dtype)
        mdm_loss = (token_nll.sum(dim=1) / l).mean()

        # (c) Sample y using f_sg(theta)
        # .detach() implements the stop-gradient: y-sampling does not back-propagate
        # through the backbone, while the MDM loss above (f_theta) does.
        prism_k = self.args.prism_k
        probs_z = F.softmax(logits_z.detach(), dim=-1)  # [b, l, V]

        y = z.clone()
        sampled_indices_mask = torch.zeros_like(masked_mask)
        for i in range(b):
            masked_positions = masked_mask[i].nonzero(as_tuple=True)[0]
            k_actual = min(prism_k, masked_positions.shape[0])
            if k_actual == 0:
                continue
            perm = torch.randperm(masked_positions.shape[0], device=input_ids.device)[:k_actual]
            selected = masked_positions[perm]
            sampled_tokens = torch.multinomial(probs_z[i, selected], num_samples=1).squeeze(-1)
            y[i, selected] = sampled_tokens
            sampled_indices_mask[i, selected] = True

        # (d) Forward pass on y
        outputs_y = model(input_ids=y, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states_y = outputs_y.hidden_states[-1]  # [b, l, d_model]

        # (e) PRISM BCE loss
        quality_scores = self.prism_head(hidden_states_y, attention_mask=attention_mask)  # [b, l]
        binary_labels = (y == input_ids).float()
        bce_loss = F.binary_cross_entropy(quality_scores, binary_labels, reduction="none")
        prism_loss = (bce_loss * sampled_indices_mask.float()).sum() / sampled_indices_mask.sum().clamp_min(1)

        total_loss = prism_loss + self.args.prism_lambda * mdm_loss

        current_step = int(self.state.global_step)
        should_log = (
            self.model.training
            and self.args.logging_steps > 0
            and current_step % self.args.logging_steps == 0
            and current_step != self._last_prism_log_step
        )
        if should_log:
            self.log({
                "prism_bce_loss": prism_loss.item(),
                "mdm_reg_loss": mdm_loss.item(),
                "prism_total_loss": total_loss.item(),
            })
            self._last_prism_log_step = current_step

        return (total_loss, outputs_z) if return_outputs else total_loss

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        """
        Save the underlying backbone in Hugging Face format plus the PRISM head.

        PRISM training wraps the backbone in a plain nn.Module so the default
        Trainer.save_model() path does not emit a reloadable config for the real
        model class. Save the inner model directly instead.
        """
        output_dir = output_dir or self.args.output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        backbone = getattr(self.model, "model", self.model)
        if not hasattr(backbone, "save_pretrained"):
            raise TypeError("PRISMTrainer.save_model expected a save_pretrained-compatible backbone")

        backbone.save_pretrained(output_path)

        if self.processing_class is not None and hasattr(self.processing_class, "save_pretrained"):
            self.processing_class.save_pretrained(output_path)

        torch.save(self.prism_head.state_dict(), output_path / "prism_head.pt")
        with open(output_path / "prism_head_config.json", "w", encoding="utf-8") as f:
            json.dump(self.prism_head.to_config_dict(), f, indent=2)
        torch.save(self.args, output_path / "training_args.bin")
