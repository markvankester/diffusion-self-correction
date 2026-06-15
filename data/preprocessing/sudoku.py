"""
Process raw Sudoku data to a flattened board array plus clue mask.
"""

import os

import numpy as np


def boards_to_strings(flat_boards: np.ndarray) -> list[str]:
    """
    Convert an (N, 81) int8 board array to a list of 81-character strings.

    Each character is the digit value of the corresponding cell (1-9).
    This is the canonical flat-string serialisation used by the tokenizer.

    Example:
        boards_to_strings(np.array([[5,3,4,...]]))
        -> ['534...']
    """
    return ["".join(str(int(v)) for v in row) for row in flat_boards]


def preprocess_sudoku(file_path):
    """
    Load a raw sudoku .npy file and return flattened boards plus clue masks.
    """
    folder = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    boards_path = os.path.join(folder, f"{base_name}-boards.npy")
    clues_path = os.path.join(folder, f"{base_name}-clues.npy")

    if os.path.exists(boards_path) and os.path.exists(clues_path):
        print(f"Loading cached preprocessed data from {folder}...")
        flat_boards = np.load(boards_path)
        flat_starting_clues = np.load(clues_path)
        print(f"Loaded {flat_boards.shape[0]} puzzles from cache.")
        return flat_boards, flat_starting_clues

    print(f"Loading data from {file_path}...")
    raw_data = np.load(file_path, mmap_mode="r")
    n_samples = raw_data.shape[0]

    print("Reshaping and extracting chunks...")
    chunks = raw_data[:, 1:].reshape(n_samples, 81, 4)

    rows = chunks[:, :, 0].astype(np.int8)
    cols = chunks[:, :, 1].astype(np.int8)
    values = chunks[:, :, 2].astype(np.int8)
    clue_flags = (chunks[:, :, 3] == 0).astype(np.int8)

    print("Constructing solved boards and clue masks...")
    solved_boards = np.zeros((n_samples, 9, 9), dtype=np.int8)
    starting_clues = np.zeros((n_samples, 9, 9), dtype=np.int8)

    puzzle_idx = np.arange(n_samples)[:, None]
    solved_boards[puzzle_idx, rows, cols] = values
    starting_clues[puzzle_idx, rows, cols] = clue_flags

    print("Flattening for the Transformer...")
    flat_boards = solved_boards.reshape(n_samples, 81)
    flat_starting_clues = starting_clues.reshape(n_samples, 81)

    print(f"Saving preprocessed data to {folder}...")
    np.save(boards_path, flat_boards)
    np.save(clues_path, flat_starting_clues)

    print(f"Done, processed {n_samples} puzzles.")
    return flat_boards, flat_starting_clues


if __name__ == "__main__":
    boards, clues = preprocess_sudoku("data/sudoku-train-data.npy")
    print(f"Boards shape: {boards.shape}")
    print(f"Clues shape:  {clues.shape}")
    print(f"Example board:\n{boards[0].reshape(9, 9)}")
