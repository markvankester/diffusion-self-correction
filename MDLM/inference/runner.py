from __future__ import annotations

import datetime
import os
from pathlib import Path

import torch

from diffusion import sample_trim
from diffusion.remdm import compute_initial_confidence
from diffusion.sampler import MDLMSampler, MDLMSamplerConfig
from diffusion.schedules import LinearAlphaScheduler

from .arithmetic_metrics import ArithmeticMetrics
from .arithmetic_display import write_arithmetic_steps_html
from .remasking_metrics import compute_remasking_metrics
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
    backplay_head: torch.nn.Module | None = None,
    max_length: int = 20,
    task_name: str = "",
    solutions: list[str] | None = None,
    show_steps: bool = False,
    steps_log_dir: str | os.PathLike | None = None,
    revisitable_regions: list[list[bool]] | None = None,
    injected_error_masks: list[list[bool]] | None = None,
    initial_confidence_texts: list[str] | None = None,
    quiet: bool = False,
) -> ArithmeticMetrics | SudokuMetrics | None:
    """Run MDLM sampling on prompts and print results."""
    scheduler = LinearAlphaScheduler()
    sampler = MDLMSampler(
        model=model,
        tokenizer=tokenizer,
        scheduler=scheduler,
        prism_head=prism_head,
        backplay_head=backplay_head,
    )

    config_params = {k: v for k, v in sampler_kwargs.items() if v is not None}
    if task_name == "sudoku" and "suppress_tokens" not in config_params:
        config_params["suppress_tokens"] = _sudoku_suppress_tokens(tokenizer)
    config_params["return_dict"] = True
    sample_config = MDLMSamplerConfig(**config_params)

    if not quiet:
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

    run_log_dir = None
    if not quiet:
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
    if not quiet:
        print(f"{'=' * 60}")

    for idx, prompt in enumerate(prompts, start=1):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_revisitable_region = (
            revisitable_regions[idx - 1]
            if revisitable_regions is not None and idx - 1 < len(revisitable_regions)
            else None
        )
        prompt_injected_error_mask = (
            injected_error_masks[idx - 1]
            if injected_error_masks is not None and idx - 1 < len(injected_error_masks)
            else None
        )
        prompt_initial_confidence_text = (
            initial_confidence_texts[idx - 1]
            if initial_confidence_texts is not None and idx - 1 < len(initial_confidence_texts)
            else None
        )
        if infill:
            output, full_answer, display_prompt_ids, revisitable_region = _run_infill(
                sampler=sampler,
                tokenizer=tokenizer,
                prompt=prompt,
                prompt_ids=prompt_ids,
                max_length=max_length,
                task_name=task_name,
                sample_config=sample_config,
                revisitable_region_override=prompt_revisitable_region,
                initial_confidence_text=prompt_initial_confidence_text,
            )
            if not quiet:
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
                revisitable_region=revisitable_region,
                injected_error_mask=prompt_injected_error_mask,
                quiet=quiet,
                model=model,
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
                revisitable_region=revisitable_region,
                quiet=quiet,
                model=model,
            )
        else:
            output = sampler.sample([prompt_ids], config=sample_config)
            generated_ids = output.sequences[0].tolist()
            answer = sample_trim(tokenizer, [generated_ids], [prompt_ids])[0]
            display_prompt_ids = prompt_ids
            full_answer = f"{prompt}{answer.strip()}"
            if not quiet:
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
                revisitable_region=_generation_region(output, len(prompt_ids)),
                quiet=quiet,
                model=model,
            )

        if show_stats and output.histories is not None and not quiet:
            _print_step_stats(output, tokenizer, display_prompt_ids, idx)

    if task_name == "sudoku" and sudoku_metrics.puzzles and run_log_dir is not None:
        metrics_path = run_log_dir / "metrics.json"
        sudoku_metrics.write(metrics_path)
        if not quiet:
            sudoku_metrics.print_summary(metrics_path)
    if task_name == "arithmetic" and arithmetic_metrics.examples and run_log_dir is not None:
        metrics_path = run_log_dir / "metrics.json"
        arithmetic_metrics.write(metrics_path)
        if not quiet:
            arithmetic_metrics.print_summary(metrics_path)

    if task_name == "sudoku":
        return sudoku_metrics
    elif task_name == "arithmetic":
        return arithmetic_metrics
    return None


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


def _sudoku_suppress_tokens(tokenizer) -> list[int]:
    return [
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
    ]


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
    revisitable_region_override: list[bool] | None = None,
    initial_confidence_text: str | None = None,
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

    if revisitable_region_override is not None:
        width = min(len(revisitable_region), len(revisitable_region_override))
        revisitable_region[:width] = [bool(v) for v in revisitable_region_override[:width]]

    infill_kwargs = {"revisitable_region": [revisitable_region]}
    if sample_config.remasking == "remdm_conf":
        confidence_ids = (
            tokenizer.encode(initial_confidence_text, add_special_tokens=False)
            if initial_confidence_text is not None
            else masked_prompt_ids
        )
        input_tensor = torch.tensor(
            [confidence_ids],
            dtype=torch.long,
            device=sampler.model.device,
        )
        attention_mask = torch.ones_like(input_tensor, dtype=torch.long)
        revisitable_tensor = torch.tensor(
            [revisitable_region],
            dtype=torch.bool,
            device=sampler.model.device,
        )
        infill_kwargs["initial_confidence"] = compute_initial_confidence(
            model=sampler.model,
            x_full=input_tensor,
            attention_mask=attention_mask,
            revisitable_region=revisitable_tensor,
        )

    output = sampler.infill(
        [masked_prompt_ids],
        config=sample_config,
        **infill_kwargs,
    )
    generated_ids = output.sequences[0].tolist()
    start, end = rhs_span
    answer_ids = _trim_at_eos(generated_ids[start:end], tokenizer)
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).replace(" ", "")
    return output, display_prefix.replace(" ", "") + answer.strip(), display_prompt_ids, revisitable_region


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
    revisitable_region: list[bool],
    injected_error_mask: list[bool] | None = None,
    quiet: bool = False,
    model=None,
) -> None:
    if task_name != "sudoku" or len(full_answer) != 81:
        return

    solution = solutions[idx - 1] if solutions and idx - 1 < len(solutions) else None
    clue_prompt = _sudoku_clue_prompt(prompt, revisitable_region)

    if solution:
        target_ids = tokenizer.encode(solution, add_special_tokens=False)
        remasking_metrics = compute_remasking_metrics(
            output=output,
            target_ids=target_ids,
            editable_mask=revisitable_region,
            injected_error_mask=injected_error_mask,
            sample_idx=0,
            mask_token_id=tokenizer.mask_token_id,
        )
        
        # Log additional metadata for hypotheses validation
        initial_confidence_corrupted = None
        initial_confidence_correct = None
        remedi_upm_confidence = None
        method_quality_score = None

        if model is not None and output.histories is not None and len(output.histories) > 0:
            init_ids = output.histories[0].to(model.device)
            attention_mask = torch.ones_like(init_ids)
            
            with torch.no_grad():
                outputs = model(init_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # UPM confidence (ReMEDI specific)
                model_conf = getattr(outputs, "confidences", None)
                if model_conf is not None:
                    remedi_upm_confidence_all = model_conf[0].cpu().tolist()
                else:
                    remedi_upm_confidence_all = None
                
                # Injected positions
                injected_positions = [pos for pos, is_err in enumerate(injected_error_mask) if is_err]
                if injected_positions:
                    corrupted_probs = []
                    correct_probs = []
                    upm_confs = []
                    
                    for pos in injected_positions:
                        # Corrupted token ID
                        corr_token_id = init_ids[0, pos].item()
                        corrupted_probs.append(probs[0, pos, corr_token_id].item())
                        
                        # Correct token ID
                        if pos < len(target_ids):
                            correct_token_id = target_ids[pos]
                            correct_probs.append(probs[0, pos, correct_token_id].item())
                            
                        # UPM confidence
                        if remedi_upm_confidence_all is not None:
                            upm_confs.append(remedi_upm_confidence_all[pos])
                    
                    if corrupted_probs:
                        initial_confidence_corrupted = sum(corrupted_probs) / len(corrupted_probs)
                    if correct_probs:
                        initial_confidence_correct = sum(correct_probs) / len(correct_probs)
                    if upm_confs:
                        remedi_upm_confidence = sum(upm_confs) / len(upm_confs)

        # Extract quality score if PRISM/Backplay quality scores are present
        if (
            output.quality_scores is not None
            and len(output.quality_scores) > 0
            and output.quality_scores[0] is not None
        ):
            first_quality = output.quality_scores[0][0]
            injected_positions = [pos for pos, is_err in enumerate(injected_error_mask) if is_err]
            if injected_positions:
                quality_vals = [first_quality[pos].item() for pos in injected_positions]
                method_quality_score = sum(quality_vals) / len(quality_vals)

        remasking_metrics["initial_confidence_corrupted"] = initial_confidence_corrupted
        remasking_metrics["initial_confidence_correct"] = initial_confidence_correct
        remasking_metrics["remedi_upm_confidence"] = remedi_upm_confidence
        remasking_metrics["method_quality_score"] = method_quality_score

        metrics.add(
            idx,
            prompt,
            solution,
            full_answer,
            editable_mask=revisitable_region,
            remasking_metrics=remasking_metrics,
        )
        if not quiet:
            print("\n  Ground truth (bold=clue):")
            print(
                render_sudoku_grid(
                    solution,
                    clue_str=clue_prompt,
                    injected_error_mask=injected_error_mask,
                )
            )
            _print_sudoku_injected_error_report(
                output_obj=output,
                prompt=prompt,
                output=full_answer,
                solution=solution,
                injected_error_mask=injected_error_mask,
            )

    if not quiet:
        print("\n  Model output (bold=clue, magenta=injected error, green=correct, red=wrong):")
        print(
            render_sudoku_grid(
                full_answer,
                clue_str=clue_prompt,
                solution_str=solution,
                injected_error_mask=injected_error_mask,
            )
        )

    if show_steps and output.histories is not None and run_log_dir is not None:
        write_sudoku_steps_html(
            output=output,
            tokenizer=tokenizer,
            prompt=clue_prompt,
            solution_str=solution,
            puzzle_idx=idx,
            html_path=run_log_dir / f"puzzle_{idx:03d}.html",
            injected_error_mask=injected_error_mask,
        )
        _save_trajectory_jsonl(
            run_log_dir=run_log_dir,
            idx=idx,
            prompt=prompt,
            expected_str=solution,
            full_answer=full_answer,
            output=output,
            tokenizer=tokenizer,
            display_prompt_ids=None,
        )


def _sudoku_clue_prompt(prompt: str, revisitable_region: list[bool]) -> str:
    if len(prompt) != 81 or len(revisitable_region) < 81:
        return prompt
    chars = []
    for pos, ch in enumerate(prompt):
        chars.append("0" if revisitable_region[pos] else ch)
    return "".join(chars)


def _print_sudoku_injected_error_report(
    output_obj,
    prompt: str,
    output: str,
    solution: str,
    injected_error_mask: list[bool] | None,
) -> None:
    if not injected_error_mask:
        return

    positions = [
        pos for pos, is_error in enumerate(injected_error_mask[:81])
        if is_error
    ]
    if not positions:
        return

    step_by_position = _first_remask_steps_by_position(output_obj, positions)

    print("\n  Injected error cells:")
    for pos in positions:
        row = pos // 9 + 1
        col = pos % 9 + 1
        initial = prompt[pos] if pos < len(prompt) else "?"
        target = solution[pos] if pos < len(solution) else "?"
        final = output[pos] if pos < len(output) else "?"
        status = "recovered" if final == target else "still wrong"
        remask_step = step_by_position.get(pos)
        remask_text = f"remasked step {remask_step}" if remask_step is not None else "not remasked"
        print(
            f"    r{row}c{col} pos={pos}: "
            f"initial={initial} target={target} final={final} "
            f"{status}, {remask_text}"
        )


def _first_remask_steps_by_position(output, positions: list[int]) -> dict[int, int]:
    if output.remask_indices is None:
        return {}

    pending = set(positions)
    first_steps: dict[int, int] = {}
    for step_idx, remask_tensor in enumerate(output.remask_indices, start=1):
        if not pending:
            break
        remasked = remask_tensor[0].bool()
        for pos in list(pending):
            if pos < remasked.numel() and bool(remasked[pos].item()):
                first_steps[pos] = step_idx
                pending.remove(pos)
    return first_steps


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
    revisitable_region: list[bool],
    quiet: bool = False,
    model=None,
) -> None:
    if task_name != "arithmetic":
        return
    
    expected_str = solutions[idx - 1] if solutions and idx - 1 < len(solutions) else None

    if expected_str is not None:
        target_ids = tokenizer.encode(expected_str, add_special_tokens=False)
        injected_error_mask = _initial_visible_error_mask(
            output=output,
            target_ids=target_ids,
            editable_mask=revisitable_region,
            mask_token_id=tokenizer.mask_token_id,
        )
        remasking_metrics = compute_remasking_metrics(
            output=output,
            target_ids=target_ids,
            editable_mask=revisitable_region,
            injected_error_mask=injected_error_mask,
            sample_idx=0,
            mask_token_id=tokenizer.mask_token_id,
        )
        
        # Log additional metadata for hypotheses validation
        # 1. Operator type
        operator = "unknown"
        for op in ["+", "-", "*"]:
            if op in prompt:
                operator = op
                break
        remasking_metrics["operator"] = operator
        
        # 2. Extract initial confidence values
        initial_confidence_corrupted = None
        initial_confidence_correct = None
        remedi_upm_confidence = None
        method_quality_score = None

        if model is not None and output.histories is not None and len(output.histories) > 0:
            init_ids = output.histories[0].to(model.device)
            attention_mask = torch.ones_like(init_ids)
            
            with torch.no_grad():
                outputs = model(init_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # UPM confidence (ReMEDI specific)
                model_conf = getattr(outputs, "confidences", None)
                if model_conf is not None:
                    remedi_upm_confidence_all = model_conf[0].cpu().tolist()
                else:
                    remedi_upm_confidence_all = None
                
                # Injected positions
                injected_positions = [pos for pos, is_err in enumerate(injected_error_mask) if is_err]
                if injected_positions:
                    corrupted_probs = []
                    correct_probs = []
                    upm_confs = []
                    
                    for pos in injected_positions:
                        # Corrupted token ID
                        corr_token_id = init_ids[0, pos].item()
                        corrupted_probs.append(probs[0, pos, corr_token_id].item())
                        
                        # Correct token ID
                        if pos < len(target_ids):
                            correct_token_id = target_ids[pos]
                            correct_probs.append(probs[0, pos, correct_token_id].item())
                            
                        # UPM confidence
                        if remedi_upm_confidence_all is not None:
                            upm_confs.append(remedi_upm_confidence_all[pos])
                    
                    if corrupted_probs:
                        initial_confidence_corrupted = sum(corrupted_probs) / len(corrupted_probs)
                    if correct_probs:
                        initial_confidence_correct = sum(correct_probs) / len(correct_probs)
                    if upm_confs:
                        remedi_upm_confidence = sum(upm_confs) / len(upm_confs)
                        
        # Extract quality score if PRISM/Backplay quality scores are present
        if (
            output.quality_scores is not None
            and len(output.quality_scores) > 0
            and output.quality_scores[0] is not None
        ):
            first_quality = output.quality_scores[0][0]
            injected_positions = [pos for pos, is_err in enumerate(injected_error_mask) if is_err]
            if injected_positions:
                quality_vals = [first_quality[pos].item() for pos in injected_positions]
                method_quality_score = sum(quality_vals) / len(quality_vals)

        remasking_metrics["initial_confidence_corrupted"] = initial_confidence_corrupted
        remasking_metrics["initial_confidence_correct"] = initial_confidence_correct
        remasking_metrics["remedi_upm_confidence"] = remedi_upm_confidence
        remasking_metrics["method_quality_score"] = method_quality_score

        metrics.add(
            idx=idx,
            prompt=prompt,
            expected=expected_str,
            output=full_answer,
            remasking_metrics=remasking_metrics,
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
        _save_trajectory_jsonl(
            run_log_dir=run_log_dir,
            idx=idx,
            prompt=prompt,
            expected_str=expected_str,
            full_answer=full_answer,
            output=output,
            tokenizer=tokenizer,
            display_prompt_ids=display_prompt_ids,
        )


def _save_trajectory_jsonl(
    run_log_dir: Path,
    idx: int,
    prompt: str,
    expected_str: str | None,
    full_answer: str,
    output,
    tokenizer,
    display_prompt_ids: list[int] | None = None,
) -> None:
    import json
    histories = output.histories
    if histories is None:
        return

    prompt_len = len(display_prompt_ids) if display_prompt_ids is not None else 0
    steps_data = []
    for step_idx in range(len(histories) - 1):
        prev_seq = histories[step_idx][0].tolist()
        curr_seq = histories[step_idx + 1][0].tolist()

        conf = None
        if output.confidences is not None and step_idx < len(output.confidences):
            conf = output.confidences[step_idx][0].tolist()

        qual = None
        if output.quality_scores is not None and step_idx < len(output.quality_scores):
            q_tensor = output.quality_scores[step_idx]
            if q_tensor is not None:
                qual = q_tensor[0].tolist()

        unmasked = []
        remasked = []
        if output.transfer_indices is not None and step_idx < len(output.transfer_indices):
            unmasked = output.transfer_indices[step_idx][0].nonzero(as_tuple=True)[0].tolist()
        if output.remask_indices is not None and step_idx < len(output.remask_indices):
            remasked = output.remask_indices[step_idx][0].nonzero(as_tuple=True)[0].tolist()

        steps_data.append({
            "step_idx": step_idx + 1,
            "tokens": [tokenizer.convert_ids_to_tokens(tid) for tid in curr_seq],
            "token_ids": curr_seq,
            "confidences": conf,
            "quality_scores": qual,
            "unmasked_indices": unmasked,
            "remasked_indices": remasked,
        })

    trace_data = {
        "example_idx": idx,
        "prompt": prompt,
        "expected": expected_str,
        "output": full_answer,
        "prompt_len": prompt_len,
        "steps": steps_data,
    }

    jsonl_path = run_log_dir / "trajectories.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace_data) + "\n")



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


def _generation_region(output, prompt_len: int) -> list[bool]:
    if output.histories is None:
        return []
    seq_len = output.histories[0].shape[1]
    return [pos >= prompt_len for pos in range(seq_len)]


def _initial_visible_error_mask(
    output,
    target_ids: list[int],
    editable_mask: list[bool],
    mask_token_id: int,
) -> list[bool]:
    if output.histories is None:
        return []

    initial = output.histories[0][0].tolist()
    seq_len = len(initial)
    mask: list[bool] = [False] * seq_len
    for pos in range(min(seq_len, len(target_ids), len(editable_mask))):
        if not editable_mask[pos]:
            continue
        token_id = initial[pos]
        if token_id != mask_token_id and token_id != target_ids[pos]:
            mask[pos] = True
    return mask
