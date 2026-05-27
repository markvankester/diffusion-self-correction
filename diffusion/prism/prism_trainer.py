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


def _quality_detection_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    active_scores = scores[mask]
    active_labels = labels[mask].bool()
    if active_labels.numel() == 0:
        zero = scores.new_tensor(0.0)
        return {
            "accuracy": zero,
            "positive_rate": zero,
            "precision": zero,
            "recall": zero,
            "f1": zero,
            "balanced_accuracy": zero,
            "pos_score_mean": zero,
            "neg_score_mean": zero,
        }

    pred = active_scores > threshold
    tp = (pred & active_labels).sum().float()
    tn = ((~pred) & (~active_labels)).sum().float()
    fp = (pred & (~active_labels)).sum().float()
    fn = ((~pred) & active_labels).sum().float()
    pos_count = active_labels.sum().float()

    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    specificity = tn / (tn + fp).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    pos_scores = active_scores[active_labels]
    neg_scores = active_scores[~active_labels]

    return {
        "accuracy": (tp + tn) / active_labels.numel(),
        "positive_rate": pos_count / active_labels.numel(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "pos_score_mean": pos_scores.mean() if pos_scores.numel() > 0 else scores.new_tensor(0.0),
        "neg_score_mean": neg_scores.mean() if neg_scores.numel() > 0 else scores.new_tensor(0.0),
    }


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
        # compute_loss() runs per micro-batch. Accumulate PRISM diagnostics and
        # emit their mean alongside Trainer's own averaged train/loss log.
        self._prism_log_sums: dict[str, float] = {}
        self._prism_log_count = 0

    def _accumulate_prism_logs(self, logs: dict[str, torch.Tensor]) -> None:
        for key, value in logs.items():
            self._prism_log_sums[key] = self._prism_log_sums.get(key, 0.0) + float(value.detach().cpu())
        self._prism_log_count += 1

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if "loss" in logs and self._prism_log_count > 0:
            logs = {
                **logs,
                **{
                    key: value / self._prism_log_count
                    for key, value in self._prism_log_sums.items()
                },
            }
            self._prism_log_sums.clear()
            self._prism_log_count = 0
        super().log(logs, *args, **kwargs)

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
        maskable_mask = self._valid_training_mask(input_ids, labels, attention_mask)

        # (a) Sample (x, z)
        t, masked_mask, z = self._sample_diffusion_mask(input_ids, maskable_mask)

        # (b) Forward pass on z
        outputs_z = model(input_ids=z, attention_mask=attention_mask, output_hidden_states=True)
        outputs_z = self._postprocess_outputs(outputs_z)
        logits_z = outputs_z.logits  # [b, l, V]

        # MDM regularization: (1/n) sum_{j: z_j=m} -log f_theta(x_j | z), averaged over batch
        # n = number of masked positions in the sequence, matching Eq. 3 / Algorithm 1 line 10.
        token_nll = F.cross_entropy(logits_z.transpose(1, 2), input_ids, reduction="none")
        token_nll = token_nll * masked_mask.to(token_nll.dtype)
        n_masked = masked_mask.sum(dim=1).clamp_min(1)
        mdm_loss = (token_nll.sum(dim=1) / n_masked).mean()

        # (c) Sample y using f_sg(theta)
        # .detach() implements the stop-gradient: y-sampling does not back-propagate
        # through the backbone, while the MDM loss above (f_theta) does.
        prism_k = self.args.prism_k
        artifact_logits_z = self._suppress_artifact_special_tokens(logits_z.detach())
        probs_z = F.softmax(artifact_logits_z, dim=-1)  # [b, l, V]

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

        if self.model.training:
            with torch.no_grad():
                metrics = _quality_detection_metrics(quality_scores, binary_labels, sampled_indices_mask)
            self._accumulate_prism_logs({
                "prism_bce_loss": prism_loss,
                "mdm_reg_loss": mdm_loss,
                "prism_total_loss": total_loss,
                "prism_quality_accuracy": metrics["accuracy"],
                "prism_quality_positive_rate": metrics["positive_rate"],
                "prism_quality_precision": metrics["precision"],
                "prism_quality_recall": metrics["recall"],
                "prism_quality_f1": metrics["f1"],
                "prism_quality_balanced_accuracy": metrics["balanced_accuracy"],
                "prism_pos_score_mean": metrics["pos_score_mean"],
                "prism_neg_score_mean": metrics["neg_score_mean"],
            })

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
