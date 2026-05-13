from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SudokuMetrics:
    puzzles: list[dict] = field(default_factory=list)
    total_exact_matches: int = 0
    total_cell_matches: int = 0
    total_cells_to_fill: int = 0

    def add(self, idx: int, prompt: str, solution: str, output: str) -> None:
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

        self.puzzles.append({
            "puzzle_idx": idx,
            "prompt": prompt,
            "solution": solution,
            "output": output,
            "board_match": board_match,
            "cell_matches": cell_matches,
            "cells_to_fill": cells_to_fill,
            "cell_accuracy": cell_matches / max(1, cells_to_fill) * 100,
        })

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
        print(f"{'=' * 60}")
