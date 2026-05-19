from __future__ import annotations

import datetime
import os
from pathlib import Path

import torch

from diffusion import sample_trim
from diffusion.sampler import MDLMSampler, MDLMSamplerConfig
from diffusion.schedules import LinearAlphaScheduler

from .arithmetic_metrics import ArithmeticMetrics
from .arithmetic_display import write_arithmetic_steps_html
from .sudoku_display import render_sudoku_grid, write_sudoku_steps_html
from .sudoku_metrics import SudokuMetrics


def run_prompts(
    prompts: list[str],
    tokenizer,
    model,
    sampler_kwargs: dict,
    show_stats: bool = False,
    infill: bool = False,
    prism_head: torch.nn.Module | None = None,
    max_length: int = 20,
    task_name: str = "",
    solutions: list[str] | None = None,
    show_steps: bool = False,
    steps_log_dir: str | os.PathLike | None = None,
) -> None:
    """Run MDLM sampling on prompts and print results."""
    scheduler = LinearAlphaScheduler()
    sampler = MDLMSampler(model=model, tokenizer=tokenizer, scheduler=scheduler, prism_head=prism_head)

    config_params = {k: v for k, v in sampler_kwargs.items() if v is not None}
    config_params["return_dict"] = True
    sample_config = MDLMSamplerConfig(**config_params)

    print(f"\n{'=' * 60}")
    print("  RUNNING INFERENCE")
    for key, value in config_params.items():
        if key == "return_dict":
            continue
        print(f"  [*] {key:15}: {value}")
    if show_stats:
        print("  [*] show_stats     : True")
    if infill:
        print("  [*] infill         : True")

    run_log_dir = _make_run_log_dir(
        task_name=task_name,
        show_steps=show_steps,
        solutions=solutions,
        steps_log_dir=steps_log_dir,
        config_params=config_params,
    )
    if show_steps and run_log_dir is not None:
        print(f"  [*] step_logs_dir  : {run_log_dir}")
    if solutions and run_log_dir is not None:
        print(f"  [*] metrics_dir    : {run_log_dir}")

    sudoku_metrics = SudokuMetrics()
    arithmetic_metrics = ArithmeticMetrics()
    print(f"{'=' * 60}")

    for idx, prompt in enumerate(prompts, start=1):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if infill:
            output, full_answer, display_prompt_ids = _run_infill(
                sampler=sampler,
                tokenizer=tokenizer,
                prompt=prompt,
                prompt_ids=prompt_ids,
                max_length=max_length,
                task_name=task_name,
                sample_config=sample_config,
            )
            print(f"\n  [{idx}] [Original] : {prompt}")
            print(f"      [Infilled] : {full_answer}")
            _handle_sudoku_result(
                idx=idx,
                prompt=prompt,
                full_answer=full_answer,
                output=output,
                tokenizer=tokenizer,
                task_name=task_name,
                solutions=solutions,
                show_steps=show_steps,
                run_log_dir=run_log_dir,
                metrics=sudoku_metrics,
            )
            _handle_arithmetic_result(
                idx=idx,
                prompt=prompt,
                full_answer=full_answer,
                output=output,
                tokenizer=tokenizer,
                task_name=task_name,
                solutions=solutions,
                show_steps=show_steps,
                run_log_dir=run_log_dir,
                metrics=arithmetic_metrics,
                display_prompt_ids=display_prompt_ids,
            )
        else:
            output = sampler.sample([prompt_ids], config=sample_config)
            generated_ids = output.sequences[0].tolist()
            answer = sample_trim(tokenizer, [generated_ids], [prompt_ids])[0]
            display_prompt_ids = prompt_ids
            full_answer = f"{prompt}{answer.strip()}"
            print(f"\n  [{idx}] {full_answer}")
            _handle_arithmetic_result(
                idx=idx,
                prompt=prompt,
                full_answer=full_answer,
                output=output,
                tokenizer=tokenizer,
                task_name=task_name,
                solutions=solutions,
                show_steps=show_steps,
                run_log_dir=run_log_dir,
                metrics=arithmetic_metrics,
                display_prompt_ids=display_prompt_ids,
            )

        if show_stats and output.histories is not None:
            _print_step_stats(output, tokenizer, display_prompt_ids, idx)

    if task_name == "sudoku" and sudoku_metrics.puzzles and run_log_dir is not None:
        metrics_path = run_log_dir / "metrics.json"
        sudoku_metrics.write(metrics_path)
        sudoku_metrics.print_summary(metrics_path)
    if task_name == "arithmetic" and arithmetic_metrics.examples and run_log_dir is not None:
        metrics_path = run_log_dir / "metrics.json"
        arithmetic_metrics.write(metrics_path)
        arithmetic_metrics.print_summary(metrics_path)


def _make_run_log_dir(
    task_name: str,
    show_steps: bool,
    solutions: list[str] | None,
    steps_log_dir: str | os.PathLike | None,
    config_params: dict,
) -> Path | None:
    if not _should_create_run_log_dir(task_name, show_steps, solutions):
        return None
    run_name = _run_name(config_params)
    log_root = Path(steps_log_dir or "./inference_logs")
    run_log_dir = log_root / _safe_name(task_name or "unknown") / run_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    return run_log_dir


def _run_name(config_params: dict) -> str:
    remasking = _safe_name(config_params.get("remasking", "none"))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{remasking}_{timestamp}"


def _safe_name(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in str(value))


def _should_create_run_log_dir(
    task_name: str,
    show_steps: bool,
    solutions: list[str] | None,
) -> bool:
    if task_name in ("sudoku", "arithmetic"):
        return show_steps or bool(solutions)
    return False


def _run_infill(
    sampler,
    tokenizer,
    prompt: str,
    prompt_ids: list[int],
    max_length: int,
    task_name: str,
    sample_config: MDLMSamplerConfig,
):
    eq_token_id = tokenizer.convert_tokens_to_ids("=")
    if eq_token_id in prompt_ids:
        eq_idx = prompt_ids.index(eq_token_id)
        n_masks = max(0, max_length - len(prompt_ids))
        masked_prompt_ids = list(prompt_ids) + [tokenizer.mask_token_id] * n_masks
        rhs_span = (eq_idx + 1, len(masked_prompt_ids))
        revisitable_region = [False] * len(masked_prompt_ids)
        for pos in range(rhs_span[0], rhs_span[1]):
            revisitable_region[pos] = True
        display_prompt_ids = prompt_ids[: eq_idx + 1]
        display_prefix = prompt[: prompt.rfind("=") + 1]
    else:
        zero_token_id = tokenizer.convert_tokens_to_ids("0")
        if task_name == "sudoku" and zero_token_id in prompt_ids:
            masked_prompt_ids = [
                tokenizer.mask_token_id if tid == zero_token_id else tid
                for tid in prompt_ids
            ]
            revisitable_region = [tid == zero_token_id for tid in prompt_ids]
            rhs_span = (0, len(masked_prompt_ids))
            display_prompt_ids = []
            display_prefix = ""
        else:
            n_masks = max(0, max_length - len(prompt_ids))
            masked_prompt_ids = prompt_ids + [tokenizer.mask_token_id] * n_masks
            revisitable_region = [False] * len(masked_prompt_ids)
            for pos in range(len(prompt_ids), len(masked_prompt_ids)):
                revisitable_region[pos] = True
            rhs_span = (len(prompt_ids), len(masked_prompt_ids))
            display_prompt_ids = prompt_ids
            display_prefix = prompt

    output = sampler.infill(
        [masked_prompt_ids],
        config=sample_config,
        revisitable_region=[revisitable_region],
    )
    generated_ids = output.sequences[0].tolist()
    start, end = rhs_span
    answer_ids = _trim_at_eos(generated_ids[start:end], tokenizer)
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).replace(" ", "")
    return output, display_prefix.replace(" ", "") + answer.strip(), display_prompt_ids


def _trim_at_eos(token_ids: list[int], tokenizer) -> list[int]:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        return token_ids
    if eos_id not in token_ids:
        return token_ids
    return token_ids[: token_ids.index(eos_id)]


def _handle_sudoku_result(
    idx: int,
    prompt: str,
    full_answer: str,
    output,
    tokenizer,
    task_name: str,
    solutions: list[str] | None,
    show_steps: bool,
    run_log_dir: Path | None,
    metrics: SudokuMetrics,
) -> None:
    if task_name != "sudoku" or len(full_answer) != 81:
        return

    solution = solutions[idx - 1] if solutions and idx - 1 < len(solutions) else None

    if solution:
        metrics.add(idx, prompt, solution, full_answer)
        print("\n  Ground truth (bold=clue):")
        print(render_sudoku_grid(solution, clue_str=prompt))

    print("\n  Model output (bold=clue, green=correct, red=wrong):")
    print(render_sudoku_grid(full_answer, clue_str=prompt, solution_str=solution))

    if show_steps and output.histories is not None and run_log_dir is not None:
        write_sudoku_steps_html(
            output=output,
            tokenizer=tokenizer,
            prompt=prompt,
            solution_str=solution,
            puzzle_idx=idx,
            html_path=run_log_dir / f"puzzle_{idx:03d}.html",
        )


def _handle_arithmetic_result(
    idx: int,
    prompt: str,
    full_answer: str,
    output,
    tokenizer,
    task_name: str,
    solutions: list[str] | None,
    show_steps: bool,
    run_log_dir: Path | None,
    metrics: ArithmeticMetrics,
    display_prompt_ids: list[int],
) -> None:
    if task_name != "arithmetic":
        return
    
    expected_str = solutions[idx - 1] if solutions and idx - 1 < len(solutions) else None

    if expected_str is not None:
        metrics.add(
            idx=idx,
            prompt=prompt,
            expected=expected_str,
            output=full_answer,
        )

    if show_steps and output.histories is not None and run_log_dir is not None:
        write_arithmetic_steps_html(
            output=output,
            tokenizer=tokenizer,
            prompt=prompt,
            expected_str=expected_str,
            task_idx=idx,
            html_path=run_log_dir / f"task_{idx:03d}.html",
            display_prompt_ids=display_prompt_ids,
        )


def _print_step_stats(output, tokenizer, display_prompt_ids: list[int], idx: int) -> None:
    print(f"\n    --- Inference Stats for Prompt [{idx}] ---")
    for step_idx in range(len(output.histories) - 1):
        curr_seq = output.histories[step_idx + 1][0].tolist()
        step_confidence = (
            output.confidences[step_idx][0]
            if output.confidences is not None and step_idx < len(output.confidences)
            else None
        )

        quality = None
        unmasked_this_step = []
        remasked_this_step = []
        if output.confidences is not None and output.transfer_indices is not None and step_idx < len(output.confidences):
            if (
                output.quality_scores is not None
                and step_idx < len(output.quality_scores)
                and output.quality_scores[step_idx] is not None
            ):
                quality = output.quality_scores[step_idx][0]
            trans_idx = output.transfer_indices[step_idx][0]
            unmasked_this_step = trans_idx.nonzero(as_tuple=True)[0].tolist()
            if output.remask_indices is not None and step_idx < len(output.remask_indices):
                remask_idx = output.remask_indices[step_idx][0]
                remasked_this_step = remask_idx.nonzero(as_tuple=True)[0].tolist()

        visual_seq = []
        for pos in range(len(display_prompt_ids), len(curr_seq)):
            token_id = curr_seq[pos]
            token_str = tokenizer.convert_ids_to_tokens(token_id)

            if pos in remasked_this_step:
                prev_token_id = output.histories[step_idx][0][pos].item()
                prev_token_str = tokenizer.convert_ids_to_tokens(prev_token_id)
                if quality is not None:
                    visual_seq.append(f"\033[91m[{prev_token_str} -> MASK (q={quality[pos].item():.2f})]\033[0m")
                else:
                    visual_seq.append(f"\033[91m[{prev_token_str} -> MASK]\033[0m")
            elif pos in unmasked_this_step and token_str != "[MASK]":
                p_val = step_confidence[pos].item() if step_confidence is not None else float("nan")
                visual_seq.append(f"\033[92m{token_str}(p={p_val:.2f})\033[0m")
            elif token_str == "[MASK]":
                visual_seq.append("\033[90m[MASK]\033[0m")
            elif quality is not None:
                visual_seq.append(f"{token_str}(q={quality[pos].item():.2f})")
            else:
                visual_seq.append(token_str)

        print(f"    Step {step_idx + 1:02d} | {' '.join(visual_seq)}")
