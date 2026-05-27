"""BackPlay trainer with look-back correction samples."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import DiffusionModelLike
from ..schedules import BaseAlphaScheduler, LinearAlphaScheduler
from ..trainer import MDLMConfig, MDLMTrainer
from .backplay_head import BackPlayHead


def _binary_detection_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    active_probs = probs[mask]
    active_labels = labels[mask].bool()
    if active_labels.numel() == 0:
        zero = probs.new_tensor(0.0)
        return {
            "accuracy": zero,
            "positive_rate": zero,
            "precision": zero,
            "recall": zero,
            "f1": zero,
            "balanced_accuracy": zero,
            "pos_prob_mean": zero,
            "neg_prob_mean": zero,
        }

    pred = active_probs > threshold
    tp = (pred & active_labels).sum().float()
    tn = ((~pred) & (~active_labels)).sum().float()
    fp = (pred & (~active_labels)).sum().float()
    fn = ((~pred) & active_labels).sum().float()
    pos_count = active_labels.sum().float()
    neg_count = (~active_labels).sum().float()

    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    specificity = tn / (tn + fp).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    pos_probs = active_probs[active_labels]
    neg_probs = active_probs[~active_labels]

    return {
        "accuracy": (tp + tn) / active_labels.numel(),
        "positive_rate": pos_count / active_labels.numel(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "pos_prob_mean": pos_probs.mean() if pos_probs.numel() > 0 else probs.new_tensor(0.0),
        "neg_prob_mean": neg_probs.mean() if neg_probs.numel() > 0 else probs.new_tensor(0.0),
    }


@dataclass
class BackPlayConfig(MDLMConfig):
    backplay_delta_t: float = 1 / 32
    backplay_loss_scope: str = "non_mask"  # "non_mask", "artifact", or "all"

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        if not (0.0 < self.backplay_delta_t < 1.0):
            raise ValueError("backplay_delta_t must be in (0, 1)")
        if self.backplay_loss_scope not in {"non_mask", "artifact", "all"}:
            raise ValueError("backplay_loss_scope must be 'non_mask', 'artifact', or 'all'")


class BackPlayTrainer(MDLMTrainer):
    """
    Train only the BackPlay correction head while keeping the DLM frozen.

    Implements the paper's look-back construction:
      1. sample a late state x_t
      2. sample a more corrupted earlier state x_{t+t'}
      3. generate artifacts from the frozen model at x_{t+t'}
      4. splice high-confidence artifacts into x_t and train error detection
    """

    def __init__(
        self,
        args: BackPlayConfig,
        backplay_head: BackPlayHead | None = None,
        *pargs,
        **kwargs,
    ):
        super().__init__(args=args, *pargs, **kwargs)
        if backplay_head is None:
            raise ValueError("backplay_head must be provided to BackPlayTrainer")
        self.backplay_head = backplay_head
        self._backplay_log_sums: dict[str, float] = {}
        self._backplay_log_count = 0

    def _accumulate_backplay_logs(self, logs: dict[str, torch.Tensor]) -> None:
        for key, value in logs.items():
            self._backplay_log_sums[key] = self._backplay_log_sums.get(key, 0.0) + float(value.detach().cpu())
        self._backplay_log_count += 1

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if "loss" in logs and self._backplay_log_count > 0:
            logs = {
                **logs,
                **{
                    key: value / self._backplay_log_count
                    for key, value in self._backplay_log_sums.items()
                },
            }
            self._backplay_log_sums.clear()
            self._backplay_log_count = 0
        super().log(logs, *args, **kwargs)

    def _sample_lbc_pair(
        self,
        model: DiffusionModelLike | nn.Module,
        input_ids: torch.Tensor,
        maskable_mask: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l = input_ids.shape
        device = input_ids.device
        delta_t = self.args.backplay_delta_t
        max_t = max(self.time_epsilon, 1.0 - delta_t)
        t = self.time_epsilon + (max_t - self.time_epsilon) * torch.rand(b, device=device)

        alpha_t = self.scheduler(t).unsqueeze(1).expand(b, l)
        xt_mask = (torch.rand((b, l), device=device) >= alpha_t) & maskable_mask
        xt = torch.where(xt_mask, self.processing_class.mask_token_id, input_ids)

        upper = (1.0 - t).clamp_min(delta_t)
        t_prime = delta_t + (upper - delta_t) * torch.rand(b, device=device)
        u = (t + t_prime).clamp_max(1.0)
        alpha_u = self.scheduler(u).unsqueeze(1).expand(b, l)
        keep_given_unmasked = (alpha_u / alpha_t.clamp_min(1e-6)).clamp(0.0, 1.0)
        extra_mask = (torch.rand((b, l), device=device) >= keep_given_unmasked) & (~xt_mask) & maskable_mask
        xu_mask = xt_mask | extra_mask
        xu = torch.where(xu_mask, self.processing_class.mask_token_id, input_ids)

        with torch.no_grad():
            outputs_u = model(input_ids=xu, attention_mask=attention_mask)
            outputs_u = self._postprocess_outputs(outputs_u)
            logits_u = self._suppress_artifact_special_tokens(outputs_u.logits)
            probs_u = F.softmax(logits_u, dim=-1)
            sampled = torch.multinomial(probs_u.reshape(-1, probs_u.size(-1)), num_samples=1).view(b, l)
            sampled_p = torch.gather(probs_u, dim=-1, index=sampled.unsqueeze(-1)).squeeze(-1)

        artifact_mask = torch.zeros_like(maskable_mask)
        for row in range(b):
            eligible = extra_mask[row] & maskable_mask[row]
            n_eligible = int(eligible.sum().item())
            if n_eligible == 0:
                continue
            valid_len = int(maskable_mask[row].sum().item())
            k = max(1, math.ceil(valid_len * delta_t))
            k_row = min(k, n_eligible)
            scores = sampled_p[row].clone()
            scores[~eligible] = -torch.inf
            _, selected = torch.topk(scores, k=k_row)
            artifact_mask[row, selected] = True

        zt = xt.clone()
        zt[artifact_mask] = sampled[artifact_mask]
        return zt, artifact_mask, t

    def compute_loss(
        self,
        model: DiffusionModelLike | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        maskable_mask = self._valid_training_mask(input_ids, labels, attention_mask)

        zt, artifact_mask, _ = self._sample_lbc_pair(model, input_ids, maskable_mask, attention_mask)

        with torch.no_grad():
            outputs_z = model(input_ids=zt, attention_mask=attention_mask, output_hidden_states=True)
            outputs_z = self._postprocess_outputs(outputs_z)
            hidden_states = outputs_z.hidden_states[self.backplay_head.config.hidden_state_index]

        error_probs = self.backplay_head(hidden_states, attention_mask=attention_mask)
        error_labels = (zt != input_ids).float()

        if self.args.backplay_loss_scope == "artifact":
            loss_mask = artifact_mask
        elif self.args.backplay_loss_scope == "non_mask":
            loss_mask = (zt != self.processing_class.mask_token_id) & maskable_mask
        else:
            loss_mask = maskable_mask

        # Class-balanced BCE: upweight the error=1 class to counteract the
        # severe imbalance where most artifact positions are correct predictions.
        # With pos_weight = n_neg / n_pos, both classes contribute equally to the loss
        # regardless of their ratio, preventing collapse to "predict 0 everywhere".
        masked_labels = error_labels[loss_mask]
        n_pos = masked_labels.sum().clamp_min(1)
        n_neg = (1 - masked_labels).sum().clamp_min(1)
        pos_weight = n_neg / n_pos  # scalar: weight for error=1 class
        # Apply per-token: error positions get pos_weight, clean positions get 1
        token_weight = torch.where(error_labels.bool(), pos_weight, torch.ones_like(error_probs))
        bce = F.binary_cross_entropy(error_probs, error_labels, reduction="none")
        effective_weight = token_weight * loss_mask.float()
        loss = (bce * effective_weight).sum() / effective_weight.sum().clamp_min(1)

        if self.model.training:
            with torch.no_grad():
                metrics = _binary_detection_metrics(error_probs, error_labels, loss_mask)
                artifact_error_rate = (
                    error_labels[artifact_mask].mean()
                    if artifact_mask.any()
                    else error_probs.new_tensor(0.0)
                )
            self._accumulate_backplay_logs(
                {
                    "backplay_bce_loss": loss,
                    "backplay_error_recall": metrics["recall"],
                    "backplay_error_f1": metrics["f1"],
                    "backplay_pos_prob_mean": metrics["pos_prob_mean"],
                    "backplay_neg_prob_mean": metrics["neg_prob_mean"],
                    "backplay_artifact_error_rate": artifact_error_rate,
                }
            )

        return (loss, outputs_z) if return_outputs else loss

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        backbone = getattr(self.model, "model", self.model)
        if not hasattr(backbone, "save_pretrained"):
            raise TypeError("BackPlayTrainer.save_model expected a save_pretrained-compatible backbone")

        backbone.save_pretrained(output_path)
        if self.processing_class is not None and hasattr(self.processing_class, "save_pretrained"):
            self.processing_class.save_pretrained(output_path)

        torch.save(self.backplay_head.state_dict(), output_path / "backplay_head.pt")
        with open(output_path / "backplay_head_config.json", "w", encoding="utf-8") as f:
            json.dump(self.backplay_head.to_config_dict(), f, indent=2)
        torch.save(self.args, output_path / "training_args.bin")
