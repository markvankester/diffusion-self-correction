"""
Generate corrupted Sudoku recovery data from an existing solved test set.

For each board, a (k, t_frac) pair is assigned via balanced round-robin over
the full experimental grid. Exactly k non-clue cells are corrupted with an
incorrect digit, then an initial recovery state is created by masking
non-clue, non-error cells with probability t_frac. Per-error and
example-level difficulty metrics are computed and saved alongside the data.

Output files (backward-compatible with existing inference code):
  - {base}-corrupted-boards.npy   (N, 81) int8 corrupted boards
  - {base}-corrupted-errors.npy   (N, 81) int8 error masks
  - {base}-initial-states.npy     (N, 81) int8 initial recovery states
  - {base}-recovery-metadata.csv  per-board difficulty metrics
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from data.preprocessing.sudoku import preprocess_sudoku


# ---------------------------------------------------------------------------
# Sudoku geometry helpers
# ---------------------------------------------------------------------------

def _cell_row(pos: int) -> int:
    return pos // 9


def _cell_col(pos: int) -> int:
    return pos % 9


def _cell_box(pos: int) -> int:
    return (_cell_row(pos) // 3) * 3 + (_cell_col(pos) // 3)


def _peers(pos: int) -> list[int]:
    """Return all peer positions (same row, col, or box) excluding pos itself."""
    r, c, b = _cell_row(pos), _cell_col(pos), _cell_box(pos)
    peers = set()
    for p in range(81):
        if p == pos:
            continue
        if _cell_row(p) == r or _cell_col(p) == c or _cell_box(p) == b:
            peers.add(p)
    return sorted(peers)


# Pre-compute peer lists once
_PEERS = [_peers(p) for p in range(81)]


def _peers_in_unit(pos: int, unit: str) -> list[int]:
    """Return peers of pos in a specific unit ('row', 'col', or 'box')."""
    r, c, b = _cell_row(pos), _cell_col(pos), _cell_box(pos)
    result = []
    for p in range(81):
        if p == pos:
            continue
        if unit == "row" and _cell_row(p) == r:
            result.append(p)
        elif unit == "col" and _cell_col(p) == c:
            result.append(p)
        elif unit == "box" and _cell_box(p) == b:
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# Error injection (unchanged from original)
# ---------------------------------------------------------------------------

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
        positions = rng.choice(eligible, size=k, replace=False)
        for pos in positions:
            orig = int(flat_boards[i, pos])
            corrupted[i, pos] = rng.choice(replacement_pool[orig])
            error_positions[i, pos] = 1

    return corrupted, error_positions


# ---------------------------------------------------------------------------
# Initial state creation
# ---------------------------------------------------------------------------

def create_initial_state(
    corrupted_board: np.ndarray,
    clues: np.ndarray,
    error_mask: np.ndarray,
    t_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create an initial recovery state from a corrupted board.

    - Clue cells remain visible.
    - Injected error cells remain visible.
    - Every other non-clue, non-error cell is replaced with 0 (mask token)
      independently with probability t_frac.

    Args:
        corrupted_board: (81,) int8 corrupted board values.
        clues:           (81,) int8 binary; 1 = clue cell.
        error_mask:      (81,) int8 binary; 1 = injected error cell.
        t_frac:          Probability of masking each free cell.
        rng:             NumPy random Generator.

    Returns:
        (81,) int8 initial state, with 0 = masked.
    """
    state = corrupted_board.copy()
    for pos in range(81):
        if clues[pos] or error_mask[pos]:
            continue
        if rng.random() < t_frac:
            state[pos] = 0
    return state


# ---------------------------------------------------------------------------
# Per-error difficulty metrics
# ---------------------------------------------------------------------------

def compute_per_error_metrics(
    initial_state: np.ndarray,
    error_mask: np.ndarray,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Compute per-error difficulty metrics.

    For each injected error cell i:
      u_i: number of Sudoku units (row, col, box) where the wrong value at i
           conflicts with another visible cell. u_i in {0, 1, 2, 3}.
      p_i: number of distinct visible peer cells with the same value as i.
      p_i_correct: p_i peers that are NOT injected errors.
      p_i_error:   p_i peers that ARE injected errors.

    Args:
        initial_state: (81,) int8 initial state (0 = masked).
        error_mask:    (81,) int8 binary; 1 = injected error cell.

    Returns:
        Tuple of (u_list, p_list, p_correct_list, p_error_list) — each a
        list with one entry per injected error, in position order.
    """
    error_positions = np.where(error_mask == 1)[0]
    error_set = set(int(p) for p in error_positions)

    u_list: list[int] = []
    p_list: list[int] = []
    p_correct_list: list[int] = []
    p_error_list: list[int] = []

    for pos in error_positions:
        pos = int(pos)
        wrong_digit = int(initial_state[pos])

        # If the error cell is somehow masked (should not happen), score 0
        if wrong_digit == 0:
            u_list.append(0)
            p_list.append(0)
            p_correct_list.append(0)
            p_error_list.append(0)
            continue

        # Track distinct conflicting peers and which units have conflicts
        conflicting_peers: set[int] = set()
        unit_has_conflict = {"row": False, "col": False, "box": False}

        for unit in ("row", "col", "box"):
            for peer in _peers_in_unit(pos, unit):
                peer_val = int(initial_state[peer])
                if peer_val == 0:
                    continue  # masked cell — no information
                if peer_val == wrong_digit:
                    unit_has_conflict[unit] = True
                    conflicting_peers.add(peer)

        u_i = sum(1 for v in unit_has_conflict.values() if v)
        p_i = len(conflicting_peers)
        p_correct = sum(1 for p in conflicting_peers if p not in error_set)
        p_error = sum(1 for p in conflicting_peers if p in error_set)

        u_list.append(u_i)
        p_list.append(p_i)
        p_correct_list.append(p_correct)
        p_error_list.append(p_error)

    return u_list, p_list, p_correct_list, p_error_list


# ---------------------------------------------------------------------------
# Example-level metrics
# ---------------------------------------------------------------------------

def compute_example_metrics(
    u_list: list[int],
    p_correct_list: list[int],
    p_error_list: list[int],
    initial_state: np.ndarray,
    clues: np.ndarray,
    error_mask: np.ndarray,
) -> dict:
    """
    Compute example-level difficulty metrics from per-error scores.

    Returns dict with:
        error_quantity, u_mean, u_min, concealed_error_count,
        p_correct_mean, error_interaction_score, masked_free_cell_count,
        difficulty
    """
    K = len(u_list)
    assert K > 0

    u_mean = float(np.mean(u_list))
    u_min = int(min(u_list))
    concealed = sum(1 for u in u_list if u == 0)
    p_correct_mean = float(np.mean(p_correct_list))
    interaction = sum(p_error_list)

    # Count masked free cells (non-clue, non-error cells with value 0)
    free = (clues == 0) & (error_mask == 0)
    masked_free = int(np.sum((initial_state == 0) & free))

    # Difficulty classification based on u_min
    if u_min >= 2:
        difficulty = "exposed"
    elif u_min == 1:
        difficulty = "partially_exposed"
    else:
        difficulty = "concealed"

    return {
        "error_quantity": K,
        "u_mean": round(u_mean, 4),
        "u_min": u_min,
        "concealed_error_count": concealed,
        "p_correct_mean": round(p_correct_mean, 4),
        "error_interaction_score": interaction,
        "masked_free_cell_count": masked_free,
        "difficulty": difficulty,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_example(
    target: np.ndarray,
    clues: np.ndarray,
    corrupted: np.ndarray,
    error_mask: np.ndarray,
    initial_state: np.ndarray,
    k: int,
) -> None:
    """
    Validate a single generated example. Raises AssertionError on failure.
    """
    # Exactly k errors injected
    assert int(error_mask.sum()) == k, (
        f"Expected {k} errors, got {int(error_mask.sum())}"
    )

    # No clue cells changed
    clue_positions = clues == 1
    assert np.all(corrupted[clue_positions] == target[clue_positions]), (
        "Clue cells were modified"
    )

    # Every error cell differs from the correct board
    error_positions = error_mask == 1
    assert np.all(corrupted[error_positions] != target[error_positions]), (
        "Some error cells match the correct board"
    )

    # Unmarked non-clue cells remain correct
    unmarked_free = (error_mask == 0) & (clues == 0)
    assert np.all(corrupted[unmarked_free] == target[unmarked_free]), (
        "Unmarked non-clue cells were modified"
    )

    # Clue cells never masked in initial_state
    assert np.all(initial_state[clue_positions] != 0), (
        "Some clue cells are masked in initial_state"
    )

    # Error cells never masked in initial_state
    assert np.all(initial_state[error_positions] != 0), (
        "Some error cells are masked in initial_state"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate corrupted Sudoku recovery dataset with difficulty metrics.",
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
        help="Directory to write output files (default: data/).",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Error counts to inject (default: 1 2 3 4 5).",
    )
    parser.add_argument(
        "--t_frac_values",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        help="Mask fractions for initial state (default: 0.2 0.3 0.4 0.5 0.6 0.7 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load source data
    print(f"Loading test data from {args.data_path}...")
    flat_boards, flat_clues = preprocess_sudoku(args.data_path)
    N = len(flat_boards)
    print(f"  Loaded {N:,} boards.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build balanced (k, t_frac) assignment via round-robin
    combos = list(product(args.k_values, args.t_frac_values))
    n_combos = len(combos)
    print(f"\nExperimental grid: {len(args.k_values)} k-values × "
          f"{len(args.t_frac_values)} t_frac values = {n_combos} combinations")
    print(f"  k_values:     {args.k_values}")
    print(f"  t_frac_values: {args.t_frac_values}")
    print(f"  Boards per combo: ~{N // n_combos:,}")

    # Pre-allocate output arrays
    all_corrupted = np.zeros((N, 81), dtype=np.int8)
    all_errors = np.zeros((N, 81), dtype=np.int8)
    all_initial = np.zeros((N, 81), dtype=np.int8)
    metadata_rows: list[dict] = []

    # Group boards by their assigned k so we can batch the injection
    k_to_indices: dict[int, list[int]] = defaultdict(list)
    board_assignments: list[tuple[int, float]] = []
    for i in range(N):
        k, t_frac = combos[i % n_combos]
        board_assignments.append((k, t_frac))
        k_to_indices[k].append(i)

    # Inject errors for each k-group
    print(f"\nInjecting errors...")
    for k in sorted(k_to_indices.keys()):
        indices = k_to_indices[k]
        group_boards = flat_boards[indices]
        group_clues = flat_clues[indices]

        corrupted, error_masks = inject_errors(group_boards, group_clues, k, rng)

        for local_i, global_i in enumerate(indices):
            all_corrupted[global_i] = corrupted[local_i]
            all_errors[global_i] = error_masks[local_i]

    print(f"  Done. Injected errors for {N:,} boards.")

    # Create initial states and compute metrics
    print(f"Creating initial states and computing metrics...")
    validation_errors = 0

    for i in range(N):
        k, t_frac = board_assignments[i]
        target = flat_boards[i]
        clues = flat_clues[i]
        corrupted = all_corrupted[i]
        error_mask = all_errors[i]

        # Create initial state
        initial_state = create_initial_state(
            corrupted, clues, error_mask, t_frac, rng,
        )
        all_initial[i] = initial_state

        # Validate
        try:
            validate_example(target, clues, corrupted, error_mask, initial_state, k)
        except AssertionError as e:
            validation_errors += 1
            if validation_errors <= 5:
                print(f"  [!] Validation error on board {i}: {e}")

        # Compute per-error metrics
        u_list, p_list, p_correct_list, p_error_list = compute_per_error_metrics(
            initial_state, error_mask,
        )

        # Compute example-level metrics
        example_metrics = compute_example_metrics(
            u_list, p_correct_list, p_error_list,
            initial_state, clues, error_mask,
        )

        metadata_rows.append({
            "board_index": i,
            "k": k,
            "t_frac": t_frac,
            "difficulty": example_metrics["difficulty"],
            "error_quantity": example_metrics["error_quantity"],
            "u_per_error": " ".join(str(u) for u in u_list),
            "p_per_error": " ".join(str(p) for p in p_list),
            "p_correct_per_error": " ".join(str(p) for p in p_correct_list),
            "p_error_per_error": " ".join(str(p) for p in p_error_list),
            "u_mean": example_metrics["u_mean"],
            "u_min": example_metrics["u_min"],
            "concealed_error_count": example_metrics["concealed_error_count"],
            "p_correct_mean": example_metrics["p_correct_mean"],
            "error_interaction_score": example_metrics["error_interaction_score"],
            "masked_free_cell_count": example_metrics["masked_free_cell_count"],
        })

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1:,}/{N:,} boards...")

    if validation_errors > 0:
        print(f"\n  [!] {validation_errors} validation error(s) detected!")
    else:
        print(f"  All {N:,} examples passed validation.")

    # Save output files
    base = os.path.splitext(os.path.basename(args.data_path))[0]
    out_prefix = os.path.join(args.output_dir, base)

    boards_out = f"{out_prefix}-corrupted-boards.npy"
    errors_out = f"{out_prefix}-corrupted-errors.npy"
    initial_out = f"{out_prefix}-initial-states.npy"
    csv_out = f"{out_prefix}-recovery-metadata.csv"

    print(f"\nSaving output files...")
    np.save(boards_out, all_corrupted)
    np.save(errors_out, all_errors)
    np.save(initial_out, all_initial)

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Total examples: {N:,}")
    print(f"  {boards_out}")
    print(f"  {errors_out}")
    print(f"  {initial_out}")
    print(f"  {csv_out}")

    # Grouped summary by (k, t_frac)
    print(f"\n{'=' * 60}")
    print("Distribution by (k, t_frac):")
    groups: dict[tuple[int, float], list[dict]] = defaultdict(list)
    for row in metadata_rows:
        groups[(row["k"], row["t_frac"])].append(row)

    for (k, t_frac) in sorted(groups.keys()):
        rows = groups[(k, t_frac)]
        n = len(rows)
        mean_u_mean = float(np.mean([r["u_mean"] for r in rows]))
        mean_u_min = float(np.mean([r["u_min"] for r in rows]))
        mean_interact = float(np.mean([r["error_interaction_score"] for r in rows]))
        mean_p_correct = float(np.mean([r["p_correct_mean"] for r in rows]))

        diff_counts = defaultdict(int)
        for r in rows:
            diff_counts[r["difficulty"]] += 1

        diff_str = ", ".join(
            f"{d}: {diff_counts[d]}"
            for d in ("exposed", "partially_exposed", "concealed")
            if diff_counts[d] > 0
        )

        print(
            f"  k={k}, t={t_frac:.1f} (n={n:,}): "
            f"u_mean={mean_u_mean:.2f}, u_min_mean={mean_u_min:.2f}, "
            f"p_correct_mean={mean_p_correct:.2f}, I_mean={mean_interact:.2f} | "
            f"{diff_str}"
        )

    # Overall difficulty distribution
    print(f"\nDifficulty distribution:")
    overall_diff = defaultdict(int)
    for row in metadata_rows:
        overall_diff[row["difficulty"]] += 1
    for d in ("exposed", "partially_exposed", "concealed"):
        count = overall_diff[d]
        pct = 100.0 * count / N
        print(f"  {d:20s}: {count:>8,} ({pct:.1f}%)")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
