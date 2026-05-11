"""
Generate corrupted Sudoku test data from an existing solved test set.

For each board, k non-clue cells are replaced with an incorrect digit
(any digit 1-9 other than the correct one). The error mask records which
positions were corrupted.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from data.preprocessing.sudoku import preprocess_sudoku


def inject_errors(
    flat_boards: np.ndarray,
    flat_clues: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inject exactly k incorrect digits into each board, avoiding clue positions.

    Args:
        flat_boards: (N, 81) int8 array of correct board values (1-9).
        flat_clues:  (N, 81) int8 binary array; 1 = starting clue cell.
        k:           Number of errors to inject per board.
        rng:         NumPy random Generator for reproducibility.

    Returns:
        corrupted:       (N, 81) int8 array with k wrong digits per board.
        error_positions: (N, 81) int8 binary array; 1 = injected error cell.
    """
    N = len(flat_boards)
    corrupted = flat_boards.copy()
    error_positions = np.zeros((N, 81), dtype=np.int8)

    # Replacement digits: for each possible correct value v (1-9),
    # pick uniformly from {1,...,9} \ {v}.
    replacement_pool = {
        v: [d for d in range(1, 10) if d != v]
        for v in range(1, 10)
    }

    for i in range(N):
        # Eligible: non-clue positions only
        eligible = np.where(flat_clues[i] == 0)[0]
        if len(eligible) < k:
            # Skip boards that don't have enough non-clue cells (rare)
            continue

        # TODO (difficulty levels): replace rng.choice with a pluggable
        # position-selector strategy:
        #   random   -> rng.choice(eligible, k)              [current]
        #   easy     -> pick cells with highest constraint score
        #               (many neighbours filled -> error is obvious)
        #   hard     -> pick cells with lowest constraint score
        #               (few neighbours filled -> error is subtle)
        # constraint_score(board, pos) = unique digits in same row+col+box
        positions = rng.choice(eligible, size=k, replace=False)
        for pos in positions:
            orig = int(flat_boards[i, pos])
            corrupted[i, pos] = rng.choice(replacement_pool[orig])
            error_positions[i, pos] = 1

    return corrupted, error_positions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate corrupted Sudoku test boards for Phase 3 evaluation.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/sudoku-test-data.npy",
        help="Base path for the test data; expects <path>-boards.npy and <path>-clues.npy.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to write corrupted board files (default: data/).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        choices=[1, 3, 5],
        help="Number of errors to inject per board (1, 3, or 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Loading test data from {args.data_path}...")
    flat_boards, flat_clues = preprocess_sudoku(args.data_path)
    print(f"  Loaded {len(flat_boards):,} boards.")

    os.makedirs(args.output_dir, exist_ok=True)

    k = args.k
    print(f"\nInjecting {k} error(s) per board...")
    corrupted, error_positions = inject_errors(flat_boards, flat_clues, k, rng)

    # Derive output prefix from the input base name
    base = os.path.splitext(os.path.basename(args.data_path))[0]
    out_prefix = os.path.join(args.output_dir, f"{base}-corrupted")

    boards_out = f"{out_prefix}-boards.npy"
    errors_out = f"{out_prefix}-errors.npy"

    np.save(boards_out, corrupted)
    np.save(errors_out, error_positions)

    n_corrupted = (error_positions.sum(axis=1) > 0).sum()
    print(f"  Saved {n_corrupted:,} corrupted boards -> {boards_out}")
    print(f"  Saved error masks                     -> {errors_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
