"""
Simple sudoku board visualizer to check if boards are processed correctly.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np


from data.preprocessing.sudoku import preprocess_sudoku


def print_board(board, clues=None, errors=None):
    """Print a sudoku board with grid lines. Bold = starting clue, Red = injected error."""
    board_2d = board.reshape(9, 9)
    if clues is not None:
        clues_2d = clues.reshape(9, 9)
    if errors is not None:
        errors_2d = errors.reshape(9, 9)

    for r in range(9):
        if r % 3 == 0 and r != 0:
            print("------+-------+------")

        row_str = ""
        for c in range(9):
            if c % 3 == 0 and c != 0:
                row_str += "| "

            val = board_2d[r, c]
            if errors is not None and errors_2d[r, c] == 1:
                row_str += f"\033[91m{val}\033[0m "
            elif clues is not None and clues_2d[r, c] == 1:
                row_str += f"\033[1m{val}\033[0m "
            else:
                row_str += f"{val} "

        print(row_str)


if __name__ == "__main__":
    train_boards, train_clues = preprocess_sudoku("data/sudoku-train-data.npy")
    test_boards, test_clues = preprocess_sudoku("data/sudoku-test-data.npy")

    corrupted_boards_path = "data/sudoku-test-data-corrupted-boards.npy"
    corrupted_errors_path = "data/sudoku-test-data-corrupted-errors.npy"
    
    corrupted_boards = None
    corrupted_errors = None
    if os.path.exists(corrupted_boards_path) and os.path.exists(corrupted_errors_path):
        corrupted_boards = np.load(corrupted_boards_path)
        corrupted_errors = np.load(corrupted_errors_path)

    print(f"\nTrain set: {len(train_boards)} puzzles")
    print(f"Test set:  {len(test_boards)} puzzles")
    if corrupted_boards is not None:
        print(f"Corrupted test set (k=3): {len(corrupted_boards)} puzzles")

    indices = np.random.choice(len(test_boards), size=3, replace=False)

    for idx in indices:
        print(f"\n{'='*40}")
        print(f"=== Test Puzzle {idx} ===")
        print(f"Flat: {test_boards[idx]}\n")
        print_board(test_boards[idx], test_clues[idx])
        print(f"\nStarting clues: {test_clues[idx].sum()}/81")

        if corrupted_boards is not None:
            print(f"\n=== Corrupted Test Puzzle {idx} ===")
            print_board(corrupted_boards[idx], test_clues[idx], corrupted_errors[idx])
            print(f"\nInjected errors: {corrupted_errors[idx].sum()}/81")
