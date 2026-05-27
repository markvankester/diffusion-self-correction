from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .remasking_metrics import RemaskingMetrics


@dataclass
class ArithmeticMetrics:
    """Container for arithmetic inference metrics.

    This intentionally starts with only full-answer exact match. Add new metrics
    by extending `add()` and the summary payload without changing runner logic.
    """

    examples: list[dict] = field(default_factory=list)
    full_answer_matches: int = 0
    remasking: RemaskingMetrics = field(default_factory=RemaskingMetrics)

    def add(
        self,
        idx: int,
        prompt: str,
        expected: str,
        output: str,
        remasking_metrics: dict | None = None,
    ) -> None:
        full_answer_match = output == expected
        self.full_answer_matches += int(full_answer_match)
        row = {
            "example_idx": idx,
            "prompt": prompt,
            "expected": expected,
            "output": output,
            "full_answer_match": full_answer_match,
        }
        if remasking_metrics is not None:
            self.remasking.add(remasking_metrics)
            row["remasking_metrics"] = remasking_metrics
        self.examples.append(row)

    def write(self, metrics_path: Path) -> None:
        total_examples = len(self.examples)
        summary = {
            "summary": {
                "total_examples": total_examples,
                "full_answer_accuracy": self.full_answer_matches / max(1, total_examples) * 100,
                "full_answer_matches": self.full_answer_matches,
                "remasking": self.remasking.summary(),
            },
            "examples": self.examples,
        }

        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def print_summary(self, metrics_path: Path | None = None) -> None:
        total_examples = len(self.examples)
        full_answer_acc = self.full_answer_matches / max(1, total_examples) * 100

        print(f"\n{'=' * 60}")
        print("  ARITHMETIC INFERENCE METRICS SUMMARY")
        print(f"  Total Examples: {total_examples}")
        print(f"  Full-Answer Accuracy: {full_answer_acc:.1f}% ({self.full_answer_matches}/{total_examples})")
        remasking = self.remasking.summary()
        if remasking["injected_error_count"]:
            injected_pct = remasking["injected_error_remasked_pct"]
            print(
                "  Injected Error Remask: "
                f"{_fmt_pct(injected_pct)} "
                f"({remasking['injected_error_remasked_count']}/{remasking['injected_error_count']}), "
                f"avg step {_fmt_num(remasking['injected_error_avg_first_remask_step'])}"
            )
        if remasking["correct_cell_opportunity_count"] or remasking["model_generated_error_count"]:
            false_pct = remasking["false_remasked_cell_pct"]
            model_pct = remasking["model_generated_error_remasked_pct"]
            print(
                "  False Remasking   : "
                f"{_fmt_pct(false_pct)} "
                f"({remasking['false_remasked_cell_count']}/{remasking['correct_cell_opportunity_count']})"
            )
            print(
                "  Model Error Remask: "
                f"{_fmt_pct(model_pct)} "
                f"({remasking['model_generated_error_remasked_count']}/{remasking['model_generated_error_count']}), "
                f"avg step {_fmt_num(remasking['model_generated_error_avg_first_remask_step'])}"
            )
        print(f"{'=' * 60}")


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def _fmt_num(value: float | int | None) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"
