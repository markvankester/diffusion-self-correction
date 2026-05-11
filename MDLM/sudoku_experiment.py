"""
sudoku_recovery_inference.py
============================
Evaluate whether an MDLM/PRISM Sudoku checkpoint can recover from an early
corrupted diffusion state.

The initial state is built as:
  - clue cells: kept fixed
  - injected error cells: kept visible as the wrong digit
  - other free cells: masked with probability 1 - alpha(t)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from data.preprocessing.sudoku import preprocess_sudoku
from diffusion.sampler import MDLMSampler, MDLMSamplerConfig
from diffusion.schedules import LinearAlphaScheduler
from MDLM.run_inference import load_model


R = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CLUE_CLR = "\033[38;5;220m"
UNCHANGED_CLR = "\033[38;5;245m"
MASK_CLR = "\033[38;5;75m"
REMASK_CLR = "\033[38;5;33m"
ERROR_CLR = "\033[38;5;203m"
NEW_FILL_CLR = "\033[38;5;82m"
REFILL_CLR = "\033[38;5;45m"
HEADER_CLR = "\033[38;5;81m"


def digit_token_ids(tokenizer) -> dict[int, int]:
    return {d: tokenizer.convert_tokens_to_ids(str(d)) for d in range(1, 10)}


def token_id_to_digit(token_id: int, tokenizer) -> int:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    return int(token) if token in {str(d) for d in range(1, 10)} else 0


def tokens_to_board(token_ids: list[int], tokenizer) -> np.ndarray:
    digits: list[int] = []
    for token_id in token_ids[:81]:
        digits.append(token_id_to_digit(int(token_id), tokenizer))
    return np.asarray(digits, dtype=np.int64)


def render_sudoku_grid(
    token_ids: list[int],
    tokenizer,
    clues: np.ndarray,
    error_mask: np.ndarray,
    free_masked: np.ndarray | None = None,
    transfer_mask: np.ndarray | None = None,
    remask_mask: np.ndarray | None = None,
    refilled_after_remask_mask: np.ndarray | None = None,
) -> str:
    lines: list[str] = []
    mask_id = tokenizer.mask_token_id

    for row in range(9):
        line = "  "
        for col in range(9):
            pos = row * 9 + col
            token_id = int(token_ids[pos])
            digit = token_id_to_digit(token_id, tokenizer)
            ch = "#" if token_id == mask_id else str(digit) if digit else "."

            if clues[pos]:
                cell = f"{CLUE_CLR}{BOLD}{ch}{R}"
            elif remask_mask is not None and remask_mask[pos]:
                cell = f"{REMASK_CLR}{BOLD}{ch}{R}"
            elif token_id == mask_id or (free_masked is not None and free_masked[pos]):
                cell = f"{MASK_CLR}{ch}{R}"
            elif refilled_after_remask_mask is not None and refilled_after_remask_mask[pos]:
                cell = f"{REFILL_CLR}{BOLD}{ch}{R}"
            elif transfer_mask is not None and transfer_mask[pos]:
                cell = f"{NEW_FILL_CLR}{BOLD}{ch}{R}"
            elif error_mask[pos]:
                cell = f"{ERROR_CLR}{BOLD}{ch}{R}"
            else:
                cell = f"{UNCHANGED_CLR}{ch}{R}"

            line += cell + " "
            if col in (2, 5):
                line += f"{DIM}|{R} "
        lines.append(line)
        if row in (2, 5):
            lines.append(f"  {DIM}------+-------+------{R}")
    return "\n".join(lines)


def visualize_example(
    output,
    local_i: int,
    idx: int,
    tokenizer,
    target: np.ndarray,
    corrupted: np.ndarray,
    clues: np.ndarray,
    error_mask: np.ndarray,
    free_masked: np.ndarray,
    t_frac: float,
) -> None:
    n_errors = int(error_mask.sum())
    err_positions = np.where(error_mask)[0].tolist()
    print(f"\n{HEADER_CLR}{BOLD}Example {idx}{R}")
    print(
        f"  t_frac={t_frac:.3f} | p(mask free)={t_frac:.3f} | "
        f"errors={n_errors} | masked_free={int(free_masked.sum())}"
    )
    if err_positions:
        err_desc = ", ".join(
            f"{p}(correct={int(target[p])}, visible={int(corrupted[p])})"
            for p in err_positions
        )
        print(f"  error positions: {err_desc}")
    print(
        f"  legend: {CLUE_CLR}{BOLD}clue{R} {UNCHANGED_CLR}unchanged{R} "
        f"{MASK_CLR}mask{R} {REMASK_CLR}{BOLD}remasked{R} "
        f"{ERROR_CLR}{BOLD}visible error{R} {NEW_FILL_CLR}{BOLD}new fill{R} "
        f"{REFILL_CLR}{BOLD}refilled after remask{R}"
    )

    histories = output.histories or []
    transfer_indices = output.transfer_indices or []
    remask_indices = output.remask_indices or []
    ever_remasked = np.zeros(81, dtype=bool)

    for step_idx, state in enumerate(histories):
        token_ids = state[local_i].tolist()
        if step_idx == 0:
            print(f"\n  {BOLD}Initial x_t{R}")
            transfer_mask = None
            remask_mask = None
            refilled_after_remask_mask = None
        else:
            transfer_mask = (
                transfer_indices[step_idx - 1][local_i, :81].detach().cpu().numpy().astype(bool)
                if step_idx - 1 < len(transfer_indices)
                else None
            )
            remask_mask = (
                remask_indices[step_idx - 1][local_i, :81].detach().cpu().numpy().astype(bool)
                if step_idx - 1 < len(remask_indices)
                else None
            )
            refilled_after_remask_mask = (
                transfer_mask & ever_remasked
                if transfer_mask is not None
                else np.zeros(81, dtype=bool)
            )
            if remask_mask is not None:
                ever_remasked |= remask_mask
            n_fill = int(transfer_mask.sum()) if transfer_mask is not None else 0
            n_remask = int(remask_mask.sum()) if remask_mask is not None else 0
            n_refill = int(refilled_after_remask_mask.sum())
            print(
                f"\n  {BOLD}Step {step_idx:02d}{R}  "
                f"filled={n_fill} remasked={n_remask} refilled_after_remask={n_refill}"
            )

        print(
            render_sudoku_grid(
                token_ids=token_ids,
                tokenizer=tokenizer,
                clues=clues,
                error_mask=error_mask,
                free_masked=free_masked if step_idx == 0 else None,
                transfer_mask=transfer_mask,
                remask_mask=remask_mask,
                refilled_after_remask_mask=refilled_after_remask_mask,
            )
        )


def inserted_error_first_recovery_steps(
    output,
    local_i: int,
    tokenizer,
    target: np.ndarray,
    error_mask: np.ndarray,
) -> list[int | None]:
    """
    For each inserted error position, return the first reverse step where the
    cell equals the solved target digit, but only if it is still correct in the
    final sampled board. Step 0 is the initial x_t and is not counted.
    """
    positions = np.where(error_mask)[0].tolist()
    if not positions or output.histories is None:
        return [None for _ in positions]

    final_ids = output.sequences[local_i].tolist()
    final_board = tokens_to_board(final_ids, tokenizer)
    steps: list[int | None] = []

    for pos in positions:
        if final_board[pos] != int(target[pos]):
            steps.append(None)
            continue

        first_step = None
        for step_idx, state in enumerate(output.histories[1:], start=1):
            token_id = int(state[local_i, pos].item())
            if token_id_to_digit(token_id, tokenizer) == int(target[pos]):
                first_step = step_idx
                break
        steps.append(first_step)

    return steps


def forward_diffuse_corrupted_board(
    corrupted: np.ndarray,
    clues: np.ndarray,
    error_mask: np.ndarray,
    tokenizer,
    t_frac: float,
    scheduler: LinearAlphaScheduler,
    generator: torch.Generator,
) -> tuple[list[int], list[bool], np.ndarray]:
    """
    Build the partially diffused Sudoku state used for recovery inference.

    Clues and injected errors are kept visible. Non-clue, non-error cells are
    independently masked according to q(x_t | x_0), where p(mask)=1-alpha(t).
    """
    alpha_t = scheduler.alpha(t_frac)
    mask_prob = 1.0 - float(alpha_t)
    mask_id = tokenizer.mask_token_id
    digit_ids = digit_token_ids(tokenizer)

    input_ids: list[int] = []
    revisitable_region: list[bool] = []
    free_masked = np.zeros(81, dtype=np.int8)

    for pos in range(81):
        if clues[pos] or error_mask[pos]:
            input_ids.append(digit_ids[int(corrupted[pos])])
        elif torch.rand((), generator=generator).item() < mask_prob:
            input_ids.append(mask_id)
            free_masked[pos] = 1
        else:
            input_ids.append(digit_ids[int(corrupted[pos])])

        revisitable_region.append(not bool(clues[pos]))

    return input_ids, revisitable_region, free_masked


def load_sudoku_arrays(
    data_path: str,
    corrupted_boards_path: str,
    error_masks_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boards, clues = preprocess_sudoku(data_path)

    corrupted_path = Path(corrupted_boards_path)
    errors_path = Path(error_masks_path)
    if not corrupted_path.exists():
        raise FileNotFoundError(f"Corrupted boards not found: {corrupted_path}")
    if not errors_path.exists():
        raise FileNotFoundError(f"Error masks not found: {errors_path}")

    corrupted = np.load(corrupted_path)
    error_masks = np.load(errors_path)
    if len(corrupted) != len(boards) or len(error_masks) != len(boards):
        raise ValueError(
            "Sudoku arrays must have matching first dimension: "
            f"boards={len(boards)}, corrupted={len(corrupted)}, errors={len(error_masks)}"
        )

    return (
        boards.astype(np.int64),
        clues.astype(np.int64),
        corrupted.astype(np.int64),
        error_masks.astype(np.int64),
    )


def evaluate_batch(
    sampler: MDLMSampler,
    tokenizer,
    scheduler: LinearAlphaScheduler,
    boards: np.ndarray,
    clues: np.ndarray,
    corrupted: np.ndarray,
    error_masks: np.ndarray,
    indices: list[int],
    args: argparse.Namespace,
) -> list[dict[str, float | int | str]]:
    sample_config = MDLMSamplerConfig(
        steps=args.steps,
        temperature=args.temperature,
        remasking=args.remasking,
        stochastic_transfer=args.stochastic_transfer,
        prism_eta=args.prism_eta,
        prism_quality_threshold=args.prism_quality_threshold,
        block_size=args.block_size,
        suppress_tokens=[
            token_id
            for token_id in (
                tokenizer.mask_token_id,
                tokenizer.bos_token_id,
                tokenizer.unk_token_id,
                tokenizer.pad_token_id,
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("0"),
            )
            if token_id is not None
        ],
        return_dict=True,
    )

    rows: list[dict[str, float | int | str]] = []
    torch_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    timestep_rng = np.random.default_rng(args.seed)

    for start in range(0, len(indices), args.batch_size):
        batch_indices = indices[start : start + args.batch_size]
        batch_inputs: list[list[int]] = []
        batch_revisitable: list[list[bool]] = []
        batch_free_masked: list[np.ndarray] = []
        batch_t_fracs = timestep_rng.uniform(
            low=args.t_frac_min,
            high=args.t_frac_max,
            size=len(batch_indices),
        )

        for local_i, idx in enumerate(batch_indices):
            t_frac = float(batch_t_fracs[local_i])
            input_ids, revisitable, free_masked = forward_diffuse_corrupted_board(
                corrupted=corrupted[idx],
                clues=clues[idx],
                error_mask=error_masks[idx],
                tokenizer=tokenizer,
                t_frac=t_frac,
                scheduler=scheduler,
                generator=torch_generator,
            )
            batch_inputs.append(input_ids)
            batch_revisitable.append(revisitable)
            batch_free_masked.append(free_masked)

        output = sampler.infill(
            batch_inputs,
            config=sample_config,
            revisitable_region=batch_revisitable,
        )

        for local_i, idx in enumerate(batch_indices):
            recovered = tokens_to_board(output.sequences[local_i].tolist(), tokenizer)
            target = boards[idx]
            clue_mask = clues[idx].astype(bool)
            error_mask = error_masks[idx].astype(bool)
            free_mask = ~clue_mask

            all_cell_accuracy = float((recovered == target).mean())
            non_clue_cell_accuracy = float((recovered[free_mask] == target[free_mask]).mean())
            inserted_error_count = int(error_mask.sum())
            inserted_error_recovery_accuracy = (
                float((recovered[error_mask] == target[error_mask]).mean())
                if inserted_error_count > 0
                else float("nan")
            )
            error_recovery_steps = inserted_error_first_recovery_steps(
                output=output,
                local_i=local_i,
                tokenizer=tokenizer,
                target=target,
                error_mask=error_mask,
            )
            recovered_error_steps = [s for s in error_recovery_steps if s is not None]
            mean_inserted_error_first_recovery_step = (
                float(np.mean(recovered_error_steps))
                if recovered_error_steps
                else float("nan")
            )
            exact_board_accuracy = float(np.array_equal(recovered, target))
            clue_cell_violation_count = int((recovered[clue_mask] != target[clue_mask]).sum())

            rows.append(
                {
                    "index": int(idx),
                    "t_frac": float(batch_t_fracs[local_i]),
                    "free_cell_mask_probability": float(1.0 - scheduler.alpha(float(batch_t_fracs[local_i]))),
                    "inserted_error_count": inserted_error_count,
                    "masked_free_cell_count": int(batch_free_masked[local_i].sum()),
                    "all_cell_accuracy": all_cell_accuracy,
                    "non_clue_cell_accuracy": non_clue_cell_accuracy,
                    "inserted_error_recovery_accuracy": inserted_error_recovery_accuracy,
                    "inserted_error_first_recovery_steps": " ".join(
                        "NA" if s is None else str(s) for s in error_recovery_steps
                    ),
                    "inserted_error_recovered_count": len(recovered_error_steps),
                    "inserted_error_recovery_step_sum": int(sum(recovered_error_steps)),
                    "mean_inserted_error_first_recovery_step": mean_inserted_error_first_recovery_step,
                    "exact_board_accuracy": exact_board_accuracy,
                    "clue_cell_violation_count": clue_cell_violation_count,
                }
            )

            if args.visualize:
                visualize_example(
                    output=output,
                    local_i=local_i,
                    idx=idx,
                    tokenizer=tokenizer,
                    target=target,
                    corrupted=corrupted[idx],
                    clues=clues[idx],
                    error_mask=error_masks[idx],
                    free_masked=batch_free_masked[local_i],
                    t_frac=float(batch_t_fracs[local_i]),
                )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sudoku recovery inference from a partially diffused corrupted state.",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory.")
    parser.add_argument("--data_path", default="data/sudoku-test-data.npy")
    parser.add_argument(
        "--corrupted_boards_path",
        default="data/sudoku-test-data-corrupted-boards.npy",
        help="Pre-generated corrupted Sudoku boards.",
    )
    parser.add_argument(
        "--error_masks_path",
        default="data/sudoku-test-data-corrupted-errors.npy",
        help="Pre-generated binary mask of inserted error positions.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num_examples", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t_frac_min", type=float, default=0.2)
    parser.add_argument("--t_frac_max", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=81)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--remasking",
        default="random",
        choices=["low_confidence", "random", "prism"],
    )
    parser.add_argument("--stochastic_transfer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prism_eta", type=float, default=0.2)
    parser.add_argument("--prism_quality_threshold", type=float, default=None)
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print Sudoku grids for the initial state and each reverse-diffusion step.",
    )
    parser.add_argument("--output_csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.t_frac_min <= args.t_frac_max <= 1.0:
        raise ValueError("Require 0 <= --t_frac_min <= --t_frac_max <= 1.")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else
        args.device
    )

    print(f"[*] Device      : {device.type.upper()}")
    print(f"[*] Checkpoint  : {args.checkpoint}")
    print(f"[*] t_frac      : uniform({args.t_frac_min}, {args.t_frac_max})")
    print(f"[*] remasking   : {args.remasking}")
    if args.visualize:
        print("[*] visualize   : True")

    tokenizer, model, prism_head = load_model(args.checkpoint, device)
    if args.remasking == "prism" and prism_head is None:
        print("[!] remasking='prism' requested, but no PRISM head was found in the checkpoint.")

    scheduler = LinearAlphaScheduler()
    sampler = MDLMSampler(
        model=model,
        tokenizer=tokenizer,
        scheduler=scheduler,
        prism_head=prism_head,
    )

    boards, clues, corrupted, error_masks = load_sudoku_arrays(
        data_path=args.data_path,
        corrupted_boards_path=args.corrupted_boards_path,
        error_masks_path=args.error_masks_path,
    )

    stop = min(args.offset + args.num_examples, len(boards))
    indices = list(range(args.offset, stop))
    if not indices:
        raise ValueError("No examples selected.")

    rows = evaluate_batch(
        sampler=sampler,
        tokenizer=tokenizer,
        scheduler=scheduler,
        boards=boards,
        clues=clues,
        corrupted=corrupted,
        error_masks=error_masks,
        indices=indices,
        args=args,
    )

    mean_non_clue_cell_accuracy = float(np.mean([r["non_clue_cell_accuracy"] for r in rows]))
    mean_inserted_error_recovery_accuracy = float(np.nanmean([r["inserted_error_recovery_accuracy"] for r in rows]))
    mean_exact_board_accuracy = float(np.mean([r["exact_board_accuracy"] for r in rows]))
    recovered_step_count = int(sum(r["inserted_error_recovered_count"] for r in rows))
    recovered_step_sum = int(sum(r["inserted_error_recovery_step_sum"] for r in rows))
    mean_inserted_error_first_recovery_step = (
        recovered_step_sum / recovered_step_count
        if recovered_step_count > 0
        else float("nan")
    )

    print("\n=== Sudoku Recovery Results ===")
    print(f"error recovery accuracy : {mean_inserted_error_recovery_accuracy:.4f}")
    if np.isnan(mean_inserted_error_first_recovery_step):
        print("mean error recovery step : n/a")
    else:
        print(f"mean error recovery step : {mean_inserted_error_first_recovery_step:.2f}")

    print(f"non-clue cell accuracy           : {mean_non_clue_cell_accuracy:.4f}")
    print(f"exact board accuracy             : {mean_exact_board_accuracy:.4f}")

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[*] Wrote per-example metrics to {out_path}")


if __name__ == "__main__":
    main()
