from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RemaskingMetrics:
    injected_error_count: int = 0
    injected_error_remasked_count: int = 0
    injected_error_first_remask_step_sum: int = 0
    false_remasked_cell_count: int = 0
    correct_cell_opportunity_count: int = 0
    model_generated_error_count: int = 0
    model_generated_error_remasked_count: int = 0
    model_generated_error_first_remask_step_sum: int = 0

    def add(self, payload: dict[str, Any]) -> None:
        self.injected_error_count += int(payload["injected_error_count"])
        self.injected_error_remasked_count += int(payload["injected_error_remasked_count"])
        self.injected_error_first_remask_step_sum += int(payload["injected_error_first_remask_step_sum"])
        self.false_remasked_cell_count += int(payload["false_remasked_cell_count"])
        self.correct_cell_opportunity_count += int(payload["correct_cell_opportunity_count"])
        self.model_generated_error_count += int(payload["model_generated_error_count"])
        self.model_generated_error_remasked_count += int(payload["model_generated_error_remasked_count"])
        self.model_generated_error_first_remask_step_sum += int(
            payload["model_generated_error_first_remask_step_sum"]
        )

    def summary(self) -> dict[str, float | int | None]:
        return {
            "injected_error_count": self.injected_error_count,
            "injected_error_remasked_count": self.injected_error_remasked_count,
            "injected_error_remasked_pct": _pct(
                self.injected_error_remasked_count,
                self.injected_error_count,
            ),
            "injected_error_avg_first_remask_step": _avg(
                self.injected_error_first_remask_step_sum,
                self.injected_error_remasked_count,
            ),
            "false_remasked_cell_count": self.false_remasked_cell_count,
            "correct_cell_opportunity_count": self.correct_cell_opportunity_count,
            "false_remasked_cell_pct": _pct(
                self.false_remasked_cell_count,
                self.correct_cell_opportunity_count,
            ),
            "model_generated_error_count": self.model_generated_error_count,
            "model_generated_error_remasked_count": self.model_generated_error_remasked_count,
            "model_generated_error_remasked_pct": _pct(
                self.model_generated_error_remasked_count,
                self.model_generated_error_count,
            ),
            "model_generated_error_avg_first_remask_step": _avg(
                self.model_generated_error_first_remask_step_sum,
                self.model_generated_error_remasked_count,
            ),
        }


def compute_remasking_metrics(
    output,
    target_ids: list[int],
    editable_mask: list[bool],
    injected_error_mask: list[bool] | None = None,
    sample_idx: int = 0,
    mask_token_id: int | None = None,
) -> dict[str, Any]:
    """Measure whether remasking targets known-bad tokens and avoids correct ones.

    Step numbers are 1-based reverse diffusion steps. The initial state is step 0
    and is never counted as a remasking/recovery step.
    """
    if output.histories is None or output.remask_indices is None:
        raise ValueError("Remasking metrics require sampler output with return_dict=True")

    histories = output.histories
    remask_indices = output.remask_indices or []
    transfer_indices = output.transfer_indices or []
    seq_len = histories[0].shape[1]
    target = _to_padded_tensor(target_ids, seq_len, histories[0].device)
    editable = _to_bool_tensor(editable_mask, seq_len, histories[0].device)
    injected = _to_bool_tensor(injected_error_mask or [], seq_len, histories[0].device)
    valid_target = torch.arange(seq_len, device=histories[0].device) < len(target_ids)
    editable = editable & valid_target
    injected = injected & editable

    first_remask_step = torch.full((seq_len,), -1, dtype=torch.long, device=histories[0].device)
    false_remasked = torch.zeros((seq_len,), dtype=torch.bool, device=histories[0].device)
    correct_opportunities = torch.zeros((seq_len,), dtype=torch.bool, device=histories[0].device)

    generated_error_first_steps: list[int] = []
    generated_error_first_remask_steps: list[int | None] = []

    for step_idx in range(1, len(histories)):
        prev_state = histories[step_idx - 1][sample_idx]
        correct_before_step = editable & (prev_state == target)
        if mask_token_id is not None:
            correct_before_step = correct_before_step & (prev_state != mask_token_id)
        correct_opportunities |= correct_before_step

        remasked_this_step = (
            remask_indices[step_idx - 1][sample_idx].bool()
            if step_idx - 1 < len(remask_indices)
            else torch.zeros((seq_len,), dtype=torch.bool, device=histories[0].device)
        )
        remasked_this_step = remasked_this_step & editable
        first_time_remasked = remasked_this_step & (first_remask_step < 0)
        first_remask_step[first_time_remasked] = step_idx
        false_remasked |= remasked_this_step & correct_before_step

        if step_idx - 1 < len(transfer_indices):
            transferred = transfer_indices[step_idx - 1][sample_idx].bool() & editable
            curr_state = histories[step_idx][sample_idx]
            generated_wrong = transferred & (curr_state != target)
            if mask_token_id is not None:
                generated_wrong = generated_wrong & (curr_state != mask_token_id)
            for pos in generated_wrong.nonzero(as_tuple=True)[0].tolist():
                generated_error_first_steps.append(step_idx)
                generated_error_first_remask_steps.append(
                    _first_later_remask_step(remask_indices, sample_idx, pos, step_idx)
                )

    injected_positions = injected.nonzero(as_tuple=True)[0]
    injected_first_steps = [
        int(first_remask_step[pos].item())
        for pos in injected_positions
        if int(first_remask_step[pos].item()) > 0
    ]
    generated_recovered_steps = [
        step for step in generated_error_first_remask_steps if step is not None
    ]

    return {
        "injected_error_count": int(injected.sum().item()),
        "injected_error_remasked_count": len(injected_first_steps),
        "injected_error_remasked_pct": _pct(len(injected_first_steps), int(injected.sum().item())),
        "injected_error_first_remask_steps": injected_first_steps,
        "injected_error_first_remask_step_sum": int(sum(injected_first_steps)),
        "injected_error_avg_first_remask_step": _avg(
            int(sum(injected_first_steps)),
            len(injected_first_steps),
        ),
        "false_remasked_cell_count": int(false_remasked.sum().item()),
        "correct_cell_opportunity_count": int(correct_opportunities.sum().item()),
        "false_remasked_cell_pct": _pct(
            int(false_remasked.sum().item()),
            int(correct_opportunities.sum().item()),
        ),
        "model_generated_error_count": len(generated_error_first_steps),
        "model_generated_error_remasked_count": len(generated_recovered_steps),
        "model_generated_error_remasked_pct": _pct(
            len(generated_recovered_steps),
            len(generated_error_first_steps),
        ),
        "model_generated_error_first_steps": generated_error_first_steps,
        "model_generated_error_first_remask_steps": generated_error_first_remask_steps,
        "model_generated_error_first_remask_step_sum": int(sum(generated_recovered_steps)),
        "model_generated_error_avg_first_remask_step": _avg(
            int(sum(generated_recovered_steps)),
            len(generated_recovered_steps),
        ),
    }


def _first_later_remask_step(
    remask_indices: list[torch.Tensor],
    sample_idx: int,
    pos: int,
    generated_step: int,
) -> int | None:
    for later_step_idx in range(generated_step + 1, len(remask_indices) + 1):
        if bool(remask_indices[later_step_idx - 1][sample_idx, pos].item()):
            return later_step_idx
    return None


def _to_padded_tensor(values: list[int], length: int, device: torch.device) -> torch.Tensor:
    out = torch.full((length,), -1, dtype=torch.long, device=device)
    if values:
        width = min(len(values), length)
        out[:width] = torch.as_tensor(values[:width], dtype=torch.long, device=device)
    return out


def _to_bool_tensor(values: list[bool], length: int, device: torch.device) -> torch.Tensor:
    out = torch.zeros((length,), dtype=torch.bool, device=device)
    if values:
        width = min(len(values), length)
        out[:width] = torch.as_tensor(values[:width], dtype=torch.bool, device=device)
    return out


def _pct(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return count / total * 100.0


def _avg(total: int, count: int) -> float | None:
    if count <= 0:
        return None
    return total / count
