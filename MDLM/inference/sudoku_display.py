from __future__ import annotations

import html as html_lib
import os
from pathlib import Path

import torch


def render_sudoku_grid(
    flat: str,
    clue_str: str | None = None,
    solution_str: str | None = None,
    injected_error_mask: list[bool] | None = None,
) -> str:
    """Format an 81-character Sudoku string as a readable 9x9 grid."""
    bold_yellow = "\033[1;33m"
    green = "\033[92m"
    red = "\033[91m"
    magenta = "\033[1;95m"
    reset = "\033[0m"

    lines = []
    for row in range(9):
        row_str = "  "
        for col in range(9):
            pos = row * 9 + col
            ch = flat[pos] if pos < len(flat) else "?"
            if ch == "0":
                ch = "."

            is_clue = clue_str is not None and pos < len(clue_str) and clue_str[pos] != "0"
            is_injected_error = (
                injected_error_mask is not None
                and pos < len(injected_error_mask)
                and injected_error_mask[pos]
            )
            if is_clue:
                cell = f"{bold_yellow}{ch}{reset}"
            elif is_injected_error:
                cell = f"{magenta}{ch}{reset}"
            elif solution_str is not None and pos < len(solution_str):
                cell = f"{green if flat[pos] == solution_str[pos] else red}{ch}{reset}"
            else:
                cell = ch

            row_str += cell + " "
            if col in (2, 5):
                row_str += "| "
        lines.append(row_str)
        if row in (2, 5):
            lines.append("  ------+-------+------")
    return "\n".join(lines)


def write_sudoku_steps_html(
    output,
    tokenizer,
    prompt: str,
    solution_str: str | None,
    puzzle_idx: int,
    html_path: str | os.PathLike,
    injected_error_mask: list[bool] | None = None,
) -> None:
    """Write the full unmasking history for a single Sudoku puzzle as HTML."""
    histories = output.histories
    if histories is None or len(histories) < 2:
        return

    mask_id = tokenizer.mask_token_id
    n_steps = len(histories) - 1

    def tid_to_char(tid: int) -> str:
        if tid == mask_id:
            return "."
        tok = tokenizer.convert_ids_to_tokens(tid)
        return tok if tok and len(tok) == 1 else "?"

    def grid_html(
        token_ids_raw: list[int],
        prev_ids_raw: list[int] | None,
        is_final: bool,
        curr_conf: torch.Tensor | None,
        curr_qual: torch.Tensor | None,
    ) -> str:
        cells_html = []
        for pos in range(81):
            tid = token_ids_raw[pos] if pos < len(token_ids_raw) else mask_id
            ch = tid_to_char(tid)
            is_mask = tid == mask_id
            is_clue = pos < len(prompt) and prompt[pos] != "0"
            is_injected_error = (
                injected_error_mask is not None
                and pos < len(injected_error_mask)
                and injected_error_mask[pos]
            )
            is_newly = (
                prev_ids_raw is not None
                and pos < len(prev_ids_raw)
                and prev_ids_raw[pos] == mask_id
                and not is_mask
            )

            if is_clue:
                css = "clue"
            elif is_injected_error and is_mask:
                css = "injected masked"
            elif is_injected_error and solution_str and pos < len(solution_str):
                css = "injected-correct" if ch == solution_str[pos] else "injected-wrong"
            elif is_injected_error:
                css = "injected"
            elif is_mask:
                css = "masked"
            elif is_final and solution_str and pos < len(solution_str):
                css = "correct" if ch == solution_str[pos] else "wrong"
            elif is_newly and solution_str and pos < len(solution_str):
                css = "new-correct" if ch == solution_str[pos] else "new-wrong"
            elif is_newly:
                css = "new-fill"
            else:
                css = "filled"

            col = pos % 9
            row = pos // 9
            border_cls = ""
            if col in (3, 6):
                border_cls += " bl"
            if row in (3, 6):
                border_cls += " bt"

            tooltip_parts = []
            subtext = ""
            if is_injected_error:
                tooltip_parts.append("Injected error cell")
                if solution_str and pos < len(solution_str):
                    tooltip_parts.append(f"Target: {solution_str[pos]}")
            if curr_conf is not None and not is_clue and (is_mask or is_newly):
                conf = curr_conf[pos].item()
                tooltip_parts.append(f"Confidence: {conf:.2f}")
                subtext = f'<span class="conf">{f"{conf:.2f}".lstrip("0")}</span>'
                if curr_qual is not None:
                    tooltip_parts.append(f"Quality: {curr_qual[pos].item():.2f}")

            tooltip = ""
            if tooltip_parts:
                tooltip = f' title="{html_lib.escape(" | ".join(tooltip_parts), quote=True)}"'
            cells_html.append(
                f'<div class="cell {css}{border_cls}"{tooltip}>{html_lib.escape(ch)}{subtext}</div>'
            )
        return '<div class="grid">' + "".join(cells_html) + "</div>"

    sections = []
    for step_idx, history in enumerate(histories):
        token_ids_raw = history[0].tolist()
        prev_ids_raw = histories[step_idx - 1][0].tolist() if step_idx > 0 else None
        is_final = step_idx == len(histories) - 1

        step_conf = None
        step_qual = None
        if step_idx > 0 and output.confidences is not None and step_idx - 1 < len(output.confidences):
            step_conf = output.confidences[step_idx - 1][0]
            if (
                output.quality_scores is not None
                and step_idx - 1 < len(output.quality_scores)
                and output.quality_scores[step_idx - 1] is not None
            ):
                step_qual = output.quality_scores[step_idx - 1][0]

        free_cells = prompt.count("0")
        n_masked = sum(1 for t in token_ids_raw[:81] if t == mask_id)
        n_unmasked = sum(
            1 for pos, tid in enumerate(token_ids_raw[:81])
            if tid != mask_id and prompt[pos] == "0"
        )
        n_newly = 0
        if prev_ids_raw is not None:
            n_newly = sum(
                1 for pos in range(81)
                if prev_ids_raw[pos] == mask_id
                and token_ids_raw[pos] != mask_id
                and prompt[pos] == "0"
            )

        label = "Initial State" if step_idx == 0 else "Final State" if is_final else f"Step {step_idx} / {n_steps}"
        fill_pct = round(100 * n_unmasked / max(1, free_cells), 1)
        stats = (
            f'<span class="stat new">+{n_newly} newly filled</span> '
            f'<span class="stat">total filled <b>{n_unmasked}</b> / {free_cells}</span> '
            f'<span class="stat">still masked <b>{n_masked}</b></span>'
        )
        if step_idx == 0:
            stats = f'<span class="stat">masked <b>{n_masked}</b> / {free_cells} free cells</span>'

        sections.append(
            f'<section class="step{" final" if is_final else ""}" id="step{step_idx}">'
            f"<h2>{label}</h2>"
            f'<div class="stats">{stats}</div>'
            f'<div class="bar-wrap"><div class="bar" style="width:{fill_pct}%"></div></div>'
            f"{grid_html(token_ids_raw[:81], prev_ids_raw, is_final, step_conf, step_qual)}"
            "</section>"
        )

    target_row = f'<p class="meta"><b>Target :</b> <code>{solution_str}</code></p>' if solution_str else ""
    steps_nav = " ".join(f'<a href="#step{i}">{i}</a>' for i in range(len(histories)))
    html = _sudoku_steps_page(
        puzzle_idx=puzzle_idx,
        prompt=prompt,
        target_row=target_row,
        n_steps=n_steps,
        steps_nav=steps_nav,
        sections="".join(sections),
    )

    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)


def _sudoku_steps_page(
    puzzle_idx: int,
    prompt: str,
    target_row: str,
    n_steps: int,
    steps_nav: str,
    sections: str,
) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Sudoku unmasking - puzzle #{puzzle_idx}</title>
<style>
  :root {{
    --bg: #ffffff;
    --surface: #f8fafc;
    --border: #e2e8f0;
    --clue: #b45309;
    --masked: #94a3b8;
    --new-fill: #10b981;
    --filled: #1e293b;
    --bar-bg: #f1f5f9;
    --bar-fill: #10b981;
    --font: 'Segoe UI', system-ui, sans-serif;
    --mono: 'Consolas', 'Fira Mono', monospace;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--filled); font-family: var(--font); padding: 2rem; }}
  h1 {{ font-size: 1.5rem; font-weight: 700; color: #0f172a; margin-bottom: 0.4rem; }}
  h2 {{ font-size: 1rem; font-weight: 600; color: #1e293b; margin-bottom: 0.6rem; }}
  .meta {{ font-size: 0.8rem; color: #64748b; margin-bottom: 0.2rem; }}
  .meta code {{ color: #475569; background: #f1f5f9; padding: 1px 4px; border-radius: 3px; font-family: var(--mono); font-size: 0.75rem; }}
  nav {{ margin: 1.2rem 0 2rem; font-size: 0.8rem; }}
  nav a {{ color: #64748b; margin-right: 6px; text-decoration: none; border: 1px solid var(--border); border-radius: 4px; padding: 2px 7px; }}
  nav a:hover {{ background: var(--surface); color: #0f172a; }}
  .legend {{ display: flex; gap: 1.2rem; flex-wrap: wrap; margin-bottom: 2rem; font-size: 0.8rem; padding: 0.7rem 1rem; background: var(--surface); border-radius: 8px; border: 1px solid var(--border); }}
  .leg {{ display: flex; align-items: center; gap: 0.4rem; }}
  .leg-box {{ width: 14px; height: 14px; border-radius: 3px; }}
  section.step {{ background: var(--bg); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1.5rem; scroll-margin-top: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
  section.step.final {{ border-color: #10b981; border-width: 2px; }}
  .stats {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.7rem; font-size: 0.8rem; }}
  .stat {{ padding: 2px 10px; border-radius: 999px; background: #f8fafc; border: 1px solid var(--border); color: #64748b; }}
  .stat b {{ color: #1e293b; }}
  .stat.new {{ border-color: var(--new-fill); color: #065f46; background: #ecfdf5; }}
  .bar-wrap {{ background: var(--bar-bg); border-radius: 999px; height: 6px; margin-bottom: 1rem; overflow: hidden; }}
  .bar {{ background: var(--bar-fill); height: 100%; border-radius: 999px; }}
  .grid {{ display: grid; grid-template-columns: repeat(9, 2.6rem); gap: 3px; width: max-content; }}
  .cell {{ width: 2.6rem; height: 2.6rem; display: flex; align-items: center; justify-content: center; border-radius: 6px; font-family: var(--mono); font-size: 1.1rem; font-weight: 600; background: #fff; border: 1px solid #f1f5f9; position: relative; }}
  .conf {{ position: absolute; bottom: 1px; right: 2px; font-size: 0.55rem; font-weight: normal; color: #94a3b8; font-family: var(--font); }}
  .cell.clue {{ color: var(--clue); background: #fffbeb; border: 1px solid #fde68a; }}
  .cell.masked {{ color: #cbd5e1; background: #f8fafc; }}
  .cell.new-fill, .cell.new-correct, .cell.correct {{ color: #065f46; background: #ecfdf5; border: 1px solid #a7f3d0; }}
  .cell.new-wrong {{ color: #9d174d; background: #fdf2f8; border: 1px solid #fbcfe8; }}
  .cell.injected, .cell.injected-wrong {{ color: #86198f; background: #fdf4ff; border: 2px solid #e879f9; box-shadow: 0 0 0 2px rgba(232,121,249,0.18); }}
  .cell.injected-correct {{ color: #047857; background: #ecfdf5; border: 2px solid #34d399; box-shadow: 0 0 0 2px rgba(52,211,153,0.18); }}
  .cell.injected.masked {{ color: #86198f; background: repeating-linear-gradient(135deg, #fdf4ff 0, #fdf4ff 6px, #fae8ff 6px, #fae8ff 12px); border: 2px solid #e879f9; }}
  .cell.filled {{ color: var(--filled); background: #fff; border: 1px solid #e2e8f0; }}
  .cell.wrong {{ color: #991b1b; background: #fef2f2; border: 1px solid #fecaca; }}
  .cell.bl {{ border-left: 2.5px solid #64748b !important; margin-left: 4px; }}
  .cell.bt {{ border-top: 2.5px solid #64748b !important; margin-top: 4px; }}
</style>
</head>
<body>
<h1>Sudoku unmasking - puzzle #{puzzle_idx}</h1>
<p class="meta"><b>Puzzle :</b> <code>{html_lib.escape(prompt)}</code></p>
{target_row}
<p class="meta"><b>Steps  :</b> {n_steps}</p>
<nav>{steps_nav}</nav>
<div class="legend">
  <div class="leg"><div class="leg-box" style="background:#fffbeb;border:1px solid #fde68a"></div> clue cell</div>
  <div class="leg"><div class="leg-box" style="background:#f8fafc;border:1px solid #e2e8f0"></div> still masked</div>
  <div class="leg"><div class="leg-box" style="background:#ecfdf5;border:1px solid #a7f3d0"></div> correct fill</div>
  <div class="leg"><div class="leg-box" style="background:#fdf2f8;border:1px solid #fbcfe8"></div> wrong new fill</div>
  <div class="leg"><div class="leg-box" style="background:#fdf4ff;border:2px solid #e879f9"></div> injected error cell</div>
</div>
{sections}
</body>
</html>
"""
