import html as html_lib
import os
from pathlib import Path
import torch

def write_arithmetic_steps_html(
    output,
    tokenizer,
    prompt: str,
    expected_str: str | None,
    task_idx: int,
    html_path: str | os.PathLike,
    display_prompt_ids: list[int],
) -> None:
    """Write the full unmasking history for a single arithmetic task as HTML."""
    histories = output.histories
    if histories is None or len(histories) < 2:
        return

    n_steps = len(histories) - 1

    def tid_to_char(tid: int) -> str:
        tok = tokenizer.convert_ids_to_tokens(tid)
        return tok if tok else "?"

    sections = []
    for step_idx in range(n_steps):
        prev_seq = histories[step_idx][0].tolist()
        curr_seq = histories[step_idx + 1][0].tolist()
        
        step_confidence = None
        if output.confidences is not None and step_idx < len(output.confidences):
            step_confidence = output.confidences[step_idx][0]

        step_quality = None
        unmasked_this_step = []
        remasked_this_step = []
        if output.confidences is not None and output.transfer_indices is not None and step_idx < len(output.confidences):
            if (
                output.quality_scores is not None
                and step_idx < len(output.quality_scores)
                and output.quality_scores[step_idx] is not None
            ):
                step_quality = output.quality_scores[step_idx][0]
            
            trans_idx = output.transfer_indices[step_idx][0]
            unmasked_this_step = trans_idx.nonzero(as_tuple=True)[0].tolist()
            
            if output.remask_indices is not None and step_idx < len(output.remask_indices):
                remask_idx = output.remask_indices[step_idx][0]
                remasked_this_step = remask_idx.nonzero(as_tuple=True)[0].tolist()

        cells_html = []
        for pos in range(len(display_prompt_ids), len(curr_seq)):
            token_id = curr_seq[pos]
            token_str = tid_to_char(token_id)
            
            is_remasked = pos in remasked_this_step
            is_unmasked = pos in unmasked_this_step and token_str != "[MASK]"
            is_mask = token_str == "[MASK]"
            
            css = "filled"
            tooltip_parts = []
            subtext = ""
            
            if is_remasked:
                css = "remasked"
                prev_token_id = prev_seq[pos]
                prev_token_str = tid_to_char(prev_token_id)
                token_str = f"{prev_token_str} &rarr; MASK"
                if step_quality is not None:
                    qual = step_quality[pos].item()
                    tooltip_parts.append(f"Quality: {qual:.2f}")
                    subtext = f'<span class="conf">q={qual:.2f}</span>'
            elif is_unmasked:
                css = "unmasked"
                if step_confidence is not None:
                    conf = step_confidence[pos].item()
                    tooltip_parts.append(f"Confidence: {conf:.2f}")
                    subtext = f'<span class="conf">p={conf:.2f}</span>'
            elif is_mask:
                css = "masked"
            else:
                if step_quality is not None:
                    qual = step_quality[pos].item()
                    tooltip_parts.append(f"Quality: {qual:.2f}")
                    subtext = f'<span class="conf">q={qual:.2f}</span>'
            
            tooltip = ""
            if tooltip_parts:
                tooltip = f' title="{html_lib.escape(" | ".join(tooltip_parts), quote=True)}"'
            
            cells_html.append(
                f'<div class="cell {css}"{tooltip}>{token_str}{subtext}</div>'
            )
        
        grid_html = '<div class="seq">' + "".join(cells_html) + "</div>"
        
        label = f"Step {step_idx + 1}"
        sections.append(
            f'<section class="step" id="step{step_idx}">'
            f"<h2>{label}</h2>"
            f"{grid_html}"
            "</section>"
        )
        
    target_row = f'<p class="meta"><b>Target :</b> <code>{expected_str}</code></p>' if expected_str else ""
    steps_nav = " ".join(f'<a href="#step{i}">{i+1}</a>' for i in range(n_steps))
    
    html = _arithmetic_steps_page(
        task_idx=task_idx,
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

def _arithmetic_steps_page(
    task_idx: int,
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
<title>Arithmetic unmasking - task #{task_idx}</title>
<style>
  :root {{
    --bg: #ffffff;
    --surface: #f8fafc;
    --border: #e2e8f0;
    --masked: #94a3b8;
    --unmasked: #10b981;
    --remasked: #ef4444;
    --filled: #1e293b;
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
  .seq {{ display: flex; flex-wrap: wrap; gap: 4px; font-family: var(--mono); font-size: 1.1rem; }}
  .cell {{ padding: 0.4rem 0.6rem; display: flex; flex-direction: column; align-items: center; justify-content: center; border-radius: 6px; background: #fff; border: 1px solid #f1f5f9; position: relative; min-width: 2.5rem; }}
  .conf {{ font-size: 0.6rem; color: #64748b; margin-top: 2px; font-family: var(--font); }}
  .cell.masked {{ color: #cbd5e1; background: #f8fafc; }}
  .cell.unmasked {{ color: #065f46; background: #ecfdf5; border: 1px solid #a7f3d0; }}
  .cell.remasked {{ color: #991b1b; background: #fef2f2; border: 1px solid #fecaca; }}
  .cell.filled {{ color: var(--filled); background: #fff; border: 1px solid #e2e8f0; }}
</style>
</head>
<body>
<h1>Arithmetic unmasking - task #{task_idx}</h1>
<p class="meta"><b>Prompt :</b> <code>{html_lib.escape(prompt)}</code></p>
{target_row}
<p class="meta"><b>Steps  :</b> {n_steps}</p>
<nav>{steps_nav}</nav>
<div class="legend">
  <div class="leg"><div class="leg-box" style="background:#f8fafc;border:1px solid #e2e8f0"></div> masked</div>
  <div class="leg"><div class="leg-box" style="background:#ecfdf5;border:1px solid #a7f3d0"></div> newly unmasked</div>
  <div class="leg"><div class="leg-box" style="background:#fef2f2;border:1px solid #fecaca"></div> remasked (with previous token)</div>
</div>
{sections}
</body>
</html>
"""
