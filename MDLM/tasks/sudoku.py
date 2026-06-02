from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from .base import TaskAdapter


class SudokuTaskAdapter(TaskAdapter):
    def __init__(self):
        super().__init__(
            name="sudoku",
            default_data_path="data/sudoku-train-data.npy",
            default_inference_dataset_path="data/sudoku-test-data.npy",
            default_tokenizer_path="./tokenizers/sudoku_char_tokenizer",
            default_prompt_delimiter=None,
            default_prompts=[],
        )

    # ------------------------------------------------------------------
    # Training dataset
    # ------------------------------------------------------------------

    def build_datasets(
        self,
        tokenizer,
        data_path: str,
        seq_len: int,
        eval_data_path: str | None = None,
        limit_data: int = 0,
        mask_until_token: str | None = None,
        eval_fraction: float = 0.0,
    ):
        """
        Load Sudoku .npy boards, apply clue-conditional labels, and return
        (train_dataset, eval_dataset).

        Clue cells are masked from the loss (label = -100); the model only
        learns to predict non-clue cells, mirroring the puzzle-solving task.

        eval_fraction: fraction of training data held out for evaluation
            (e.g. 0.05 = 5 %).  Ignored if eval_fraction <= 0.
        mask_until_token: not used for Sudoku (ignored silently).
        """
        from data.preprocessing.sudoku import preprocess_sudoku
        from data.processing.sudoku_dataset import build_sudoku_hf_dataset

        flat_boards, flat_clues = preprocess_sudoku(data_path)

        N = len(flat_boards)
        if limit_data and limit_data > 0:
            N = min(N, limit_data)
            flat_boards = flat_boards[:N]
            flat_clues = flat_clues[:N]

        if eval_fraction > 0.0:
            split_idx = int(N * (1.0 - eval_fraction))
            train_dataset = build_sudoku_hf_dataset(
                flat_boards[:split_idx], flat_clues[:split_idx], tokenizer, seq_len
            )
            eval_dataset = build_sudoku_hf_dataset(
                flat_boards[split_idx:], flat_clues[split_idx:], tokenizer, seq_len
            )
            print(f"  {len(train_dataset):,} training boards, {len(eval_dataset):,} eval boards (seq_len={seq_len})")
        else:
            train_dataset = build_sudoku_hf_dataset(flat_boards, flat_clues, tokenizer, seq_len)
            eval_dataset = None
            print(f"  {len(train_dataset):,} training boards (seq_len={seq_len}), no eval split")

        return train_dataset, eval_dataset

    # ------------------------------------------------------------------
    # Inspection / inference helpers
    # ------------------------------------------------------------------

    def load_examples(self, data_path: str, offset: int, limit: int) -> list[str]:
        """Return a slice of boards as 81-character strings."""
        from data.preprocessing.sudoku import preprocess_sudoku, boards_to_strings

        flat_boards, _ = preprocess_sudoku(data_path)
        slice_ = flat_boards[offset: offset + limit]
        return boards_to_strings(slice_)

    def load_dataset_prompts(
        self,
        path: str,
        mode: str,
        num: int,
        delimiter: str | None = None,
    ) -> list[str]:
        """
        Return unsolved Sudoku puzzles for inference.

        Clue cells retain their digit; non-clue cells are set to 0 so the
        infill sampler can replace them with [MASK] before passing to the model.
        """
        from data.preprocessing.sudoku import preprocess_sudoku, boards_to_strings

        flat_boards, flat_clues = preprocess_sudoku(path)
        N = len(flat_boards)

        if num <= 0:
            num = N

        if mode == "random":
            indices = random.sample(range(N), min(num, N))
        else:
            indices = list(range(min(num, N)))

        recovery = self._load_recovery_arrays(path, N)
        if recovery is not None:
            corrupted, error_masks, initial_states = recovery
            selected_boards = flat_boards[indices]
            selected_clues = flat_clues[indices]
            selected_initial = initial_states[indices]
            selected_corrupted = corrupted[indices]
            selected_errors = error_masks[indices]

            self._solution_strings = boards_to_strings(selected_boards)
            self._revisitable_regions = [
                [not bool(v) for v in row] for row in selected_clues
            ]
            self._injected_error_masks = [
                [bool(v) for v in row] for row in selected_errors
            ]
            self._initial_confidence_strings = boards_to_strings(selected_corrupted)
            self._sudoku_recovery_indices = indices
            return boards_to_strings(selected_initial)

        selected_boards = flat_boards[indices]
        selected_clues  = flat_clues[indices]

        # Non-clue cells become 0; clue cells keep their digit value.
        puzzles = selected_boards * selected_clues

        # Store solved boards for display in run_inference (ground truth grids).
        self._solution_strings = boards_to_strings(selected_boards)
        self._revisitable_regions = [
            [bool(v == 0) for v in row] for row in puzzles
        ]
        self._injected_error_masks = None
        self._initial_confidence_strings = None
        self._sudoku_recovery_indices = indices

        return boards_to_strings(puzzles)

    def _load_recovery_arrays(
        self,
        path: str,
        expected_len: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        base_path = Path(path)
        data_dir = base_path.parent
        stem = base_path.stem
        corrupted_path = data_dir / f"{stem}-corrupted-boards.npy"
        errors_path = data_dir / f"{stem}-corrupted-errors.npy"
        initial_path = data_dir / f"{stem}-initial-states.npy"

        if not (
            corrupted_path.exists()
            and errors_path.exists()
            and initial_path.exists()
        ):
            return None

        corrupted = np.load(corrupted_path)
        error_masks = np.load(errors_path)
        initial_states = np.load(initial_path)
        if (
            len(corrupted) != expected_len
            or len(error_masks) != expected_len
            or len(initial_states) != expected_len
        ):
            raise ValueError(
                "Sudoku recovery arrays must match the base dataset length: "
                f"boards={expected_len}, corrupted={len(corrupted)}, "
                f"errors={len(error_masks)}, initial_states={len(initial_states)}"
            )
        return corrupted, error_masks, initial_states


    def describe_example(self, text: str) -> list[tuple[str, str]]:
        """Show the board as a 9×9 grid for human-readable inspection."""
        if len(text) != 81:
            return []
        rows = [text[r * 9: r * 9 + 9] for r in range(9)]
        return [("board", "\n  ".join(rows))]
