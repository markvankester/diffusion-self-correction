from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .remasking_metrics import RemaskingMetrics


@dataclass
class SudokuMetrics:
    puzzles: list[dict] = field(default_factory=list)
    total_exact_matches: int = 0
    total_cell_matches: int = 0
    total_cells_to_fill: int = 0
    remasking: RemaskingMetrics = field(default_factory=RemaskingMetrics)

    def add(
        self,
        idx: int,
        prompt: str,
        solution: str,
        output: str,
        remasking_metrics: dict | None = None,
    ) -> None:
        board_match = output == solution
        cell_matches = 0
        cells_to_fill = 0

        for pos in range(81):
            if prompt[pos] == "0":
                cells_to_fill += 1
                cell_matches += int(output[pos] == solution[pos])

        self.total_exact_matches += int(board_match)
        self.total_cell_matches += cell_matches
        self.total_cells_to_fill += cells_to_fill

        row = {
            "puzzle_idx": idx,
            "prompt": prompt,
            "solution": solution,
            "output": output,
            "board_match": board_match,
            "cell_matches": cell_matches,
            "cells_to_fill": cells_to_fill,
            "cell_accuracy": cell_matches / max(1, cells_to_fill) * 100,
        }
        if remasking_metrics is not None:
            self.remasking.add(remasking_metrics)
            row["remasking_metrics"] = remasking_metrics
        self.puzzles.append(row)

    def write(self, metrics_path: Path) -> None:
        total_puzzles = len(self.puzzles)
        board_acc = self.total_exact_matches / max(1, total_puzzles) * 100
        cell_acc = self.total_cell_matches / max(1, self.total_cells_to_fill) * 100

        summary = {
            "summary": {
                "total_puzzles": total_puzzles,
                "board_accuracy": board_acc,
                "cell_accuracy": cell_acc,
                "total_exact_matches": self.total_exact_matches,
                "total_cell_matches": self.total_cell_matches,
                "total_cells_to_fill": self.total_cells_to_fill,
                "remasking": self.remasking.summary(),
            },
            "puzzles": self.puzzles,
        }

        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def print_summary(self, metrics_path: Path | None = None) -> None:
        total_puzzles = len(self.puzzles)
        board_acc = self.total_exact_matches / max(1, total_puzzles) * 100
        cell_acc = self.total_cell_matches / max(1, self.total_cells_to_fill) * 100

        print(f"\n{'=' * 60}")
        print("  SUDOKU INFERENCE METRICS SUMMARY")
        print(f"  Total Puzzles: {total_puzzles}")
        print(f"  Board-Level Accuracy: {board_acc:.1f}% ({self.total_exact_matches}/{total_puzzles})")
        print(f"  Cell-Level Accuracy : {cell_acc:.1f}% ({self.total_cell_matches}/{self.total_cells_to_fill})")
        remasking = self.remasking.summary()
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
