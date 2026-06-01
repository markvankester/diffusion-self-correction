from typing import Any, Optional
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

from ..trainer import MDLMTrainer, MDLMConfig
from ..interfaces import DiffusionModelLike, ModelOutputLike


@dataclass
class RemeDiTrainerConfig(MDLMConfig):
    lambda_ups: float = 1.0
    r_incorrect: float = 0.1


class RemeDiTrainer(MDLMTrainer):
    def __init__(
        self,
        args: RemeDiTrainerConfig,
        *pargs,
        **kwargs,
    ):
        super().__init__(args=args, *pargs, **kwargs)
        self.lambda_ups = args.lambda_ups
        self.r_incorrect = args.r_incorrect

    def compute_loss(
        self,
        model: DiffusionModelLike | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute the RemeDi (TPS + UPS) loss following Algorithm 1 (Remask SFT).
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

        # === 1. Sample diffusion timesteps t ∈ [ε, 1] ===
        t = self.time_epsilon + (1.0 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )  # [b]

        # === 2. Compute masking/corruption probabilities ===
        p_mask = t.unsqueeze(1).expand(b, l)  # [b, l]
        p_incorrect = 4.0 * self.r_incorrect * (t * (1.0 - t)).unsqueeze(1).expand(b, l)  # [b, l]

        # === 3. Determine token type splits ===
        rand_vals_mask = torch.rand((b, l), device=input_ids.device)
        rand_vals_incorrect = torch.rand((b, l), device=input_ids.device)

        Smask = (rand_vals_mask < p_mask) & maskable_mask
        Sincorrect = (~Smask) & (rand_vals_incorrect < p_incorrect) & maskable_mask
        Sclean = (~Smask) & (~Sincorrect) & maskable_mask

        # === 4. Construct noisy/corrupted input ===
        noised_input_ids = input_ids.clone()
        noised_input_ids[Smask] = self.processing_class.mask_token_id

        # Replace incorrect positions with random alternative tokens
        random_tokens = torch.randint(
            0, self.processing_class.vocab_size, (b, l), device=input_ids.device
        )
        # Avoid correct tokens
        is_same = (random_tokens == input_ids)
        if is_same.any():
            random_tokens[is_same] = (random_tokens[is_same] + 1) % self.processing_class.vocab_size
        
        # Avoid special tokens (pad, mask, eos, bos)
        special_ids = {
            self.processing_class.pad_token_id,
            self.processing_class.mask_token_id,
            getattr(self.processing_class, "eos_token_id", None),
            getattr(self.processing_class, "bos_token_id", None),
        }
        special_ids = {sid for sid in special_ids if sid is not None}
        for sid in special_ids:
            is_special = (random_tokens == sid)
            if is_special.any():
                random_tokens[is_special] = (random_tokens[is_special] + 1) % self.processing_class.vocab_size
                is_same_again = (random_tokens == input_ids)
                if is_same_again.any():
                    random_tokens[is_same_again] = (random_tokens[is_same_again] + 1) % self.processing_class.vocab_size

        noised_input_ids[Sincorrect] = random_tokens[Sincorrect]

        # === 5. Forward pass ===
        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits
        confidences = getattr(outputs, "confidences", None)

        # === 6. Compute Diffusion Loss (only on Smask positions) ===
        loss_weights = self._compute_loss_weights(t=t, inputs=inputs)
        
        token_nll_raw = F.cross_entropy(
            logits.transpose(1, 2),
            input_ids,
            reduction="none",
        )
        token_nll = token_nll_raw * loss_weights * Smask.to(token_nll_raw.dtype)

        if self.loss_norm_type == "token":
            L_diffusion = token_nll.sum() / Smask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            L_diffusion = (token_nll.sum(-1) / Smask.sum(-1).clamp_min(1)).mean()
        elif self.loss_norm_type == "batch":
            L_diffusion = token_nll.sum() / b
        else:
            raise ValueError("Invalid loss_norm_type.")

        # === 7. Compute UPS BCE Loss ===
        if confidences is not None:
            probs = F.softmax(logits, dim=-1)
            probs_x0 = torch.gather(probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

            # Assign targets y:
            # - Clean positions: y = 1.0
            # - Incorrect positions: y = 0.0
            # - Mask positions: y = stopgrad(p_x0)
            y = torch.zeros((b, l), device=logits.device, dtype=logits.dtype)
            y[Sclean] = 1.0
            y[Sincorrect] = 0.0
            y[Smask] = probs_x0[Smask].detach()

            # BCE loss with logits
            bce_loss = F.binary_cross_entropy_with_logits(confidences, y, reduction="none")

            valid_ups_mask = maskable_mask
            if attention_mask is not None:
                valid_ups_mask = valid_ups_mask & attention_mask.bool()

            L_ups = (bce_loss * valid_ups_mask.to(bce_loss.dtype)).sum() / valid_ups_mask.sum().clamp_min(1)
        else:
            L_ups = torch.tensor(0.0, device=logits.device)

        loss = L_diffusion + self.lambda_ups * L_ups

        # === 8. Log MDLM-specific diagnostics ===
        if self.model.training and _WANDB_AVAILABLE and wandb.run is not None:
            n_masked = Smask.sum().item()
            with torch.no_grad():
                raw_nll = (token_nll_raw.detach() * Smask.to(token_nll_raw.dtype)).sum() / max(n_masked, 1)
                correct = (logits.detach().argmax(dim=-1) == input_ids) & Smask
                accuracy = correct.sum().item() / max(n_masked, 1)
            wandb.log(
                {
                    "train/unweighted_nll": raw_nll.item(),
                    "train/masked_token_accuracy": accuracy,
                    "train/L_diffusion": L_diffusion.item(),
                    "train/L_ups": L_ups.item(),
                },
                step=self.state.global_step,
            )

        return (loss, outputs) if return_outputs else loss
