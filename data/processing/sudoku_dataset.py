"""
Sudoku dataset builder for the MDLM training pipeline.

Converts pre-processed (N, 81) board arrays into a HuggingFace Dataset
with per-position clue-conditional labels:
  - Clue cells     -> label = -100 (excluded from diffusion loss)
  - Non-clue cells -> label = token_id of the correct digit
"""

from __future__ import annotations

import numpy as np
from datasets import Dataset


def build_sudoku_hf_dataset(
    flat_boards: np.ndarray,
    flat_clues: np.ndarray,
    tokenizer,
    seq_len: int,
    chunk_size: int = 50_000,
) -> Dataset:
    """
    Args:
        flat_boards:  (N, 81) int8 array of cell values (1-9).
        flat_clues:   (N, 81) int8 binary array; 1 = starting-clue cell.
        tokenizer:    Tokenizer with "1"..."9" in vocabulary.
        seq_len:      Target sequence length (should be 81 Sudoku cells).
        chunk_size:   Number of boards processed per Arrow chunk to keep
                      peak memory manageable.

    Returns:
        A HuggingFace Dataset with columns 'input_ids' and 'labels'.
    """
    n_boards = len(flat_boards)

    # Build a digit->token_id lookup table (indices 0-9).
    digit_to_token = np.array(
        [tokenizer.convert_tokens_to_ids(str(d)) for d in range(10)],
        dtype=np.int32,
    )

    # Number of Sudoku cell positions to use.
    n_cells = min(81, seq_len)

    def gen():
        for start in range(0, n_boards, chunk_size):
            end = min(start + chunk_size, n_boards)
            boards_chunk = flat_boards[start:end].astype(np.int64)  # (C, 81)
            clues_chunk = flat_clues[start:end].astype(bool)        # (C, 81)

            # Digit -> token ID conversion (vectorized).
            token_matrix = digit_to_token[boards_chunk[:, :n_cells]]  # (C, n_cells)

            # Clue-conditional labels: copy token IDs, mask clue positions with -100.
            labels_matrix = token_matrix.copy()
            labels_matrix[:, :n_cells][clues_chunk[:, :n_cells]] = -100

            # Yield each row as a dictionary.
            for i in range(len(boards_chunk)):
                yield {
                    "input_ids": token_matrix[i].tolist(),
                    "labels": labels_matrix[i].tolist(),
                }

    return Dataset.from_generator(gen)
