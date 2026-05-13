from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ArithmeticMetrics:
    """Container for arithmetic inference metrics.

    This intentionally starts with only full-answer exact match. Add new metrics
    by extending `add()` and the summary payload without changing runner logic.
    """

    examples: list[dict] = field(default_factory=list)
    full_answer_matches: int = 0

    def add(self, idx: int, prompt: str, expected: str, output: str) -> None:
        full_answer_match = output == expected
        self.full_answer_matches += int(full_answer_match)
        self.examples.append({
            "example_idx": idx,
            "prompt": prompt,
            "expected": expected,
            "output": output,
            "full_answer_match": full_answer_match,
        })

    def write(self, metrics_path: Path) -> None:
        total_examples = len(self.examples)
        summary = {
            "summary": {
                "total_examples": total_examples,
                "full_answer_accuracy": self.full_answer_matches / max(1, total_examples) * 100,
                "full_answer_matches": self.full_answer_matches,
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
        print(f"{'=' * 60}")
