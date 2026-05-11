"""
Generate arithmetic datasets following specific rules.
"""

import argparse
import json
import os
import random

# Operand ranges by digit count
_RANGES = {1: (1, 9), 2: (10, 99), 3: (100, 999), 4: (1000, 9999)}
MAX_EQ_CHARS = 19


def _pick_operand_size():
    """Pick a digit count (1-4) with bias towards harder problems.

    Distribution: 1-digit 10%, 2-digit 20%, 3-digit 35%, 4-digit 35%.
    """
    return random.choices([1, 2, 3, 4], weights=[10, 20, 35, 35])[0]


def _generate_edge_case():
    """Generate an edge-case equation to teach special arithmetic patterns."""
    case = random.choice([
        "add_zero",        # x + 0 = x
        "sub_zero",        # x - 0 = x
        "mul_one",         # x * 1 = x  or  1 * x = x
        "mul_zero",        # x * 0 = 0  or  0 * x = 0
        "self_sub",        # x - x = 0
        "carry_boundary",  # 99 + 3 = 102  (carry propagation)
        "zero_first",      # 0 + x = x
    ])

    if case == "add_zero":
        a = random.randint(1, 999)
        return f"{a}+0={a}"

    if case == "sub_zero":
        a = random.randint(1, 999)
        return f"{a}-0={a}"

    if case == "mul_one":
        a = random.randint(1, 999)
        return f"{a}*1={a}" if random.random() < 0.5 else f"1*{a}={a}"

    if case == "mul_zero":
        a = random.randint(1, 999)
        return f"{a}*0=0" if random.random() < 0.5 else f"0*{a}=0"

    if case == "self_sub":
        a = random.randint(1, 999)
        return f"{a}-{a}=0"

    if case == "carry_boundary":
        bases = [9, 19, 29, 49, 99, 199, 299, 499, 999]
        a = random.choice(bases)
        b = random.randint(1, min(10, 999 - a) if a < 999 else 1)
        return f"{a}+{b}={a + b}"

    # zero_first: 0 + x = x
    a = random.randint(1, 999)
    return f"0+{a}={a}"


def generate_equation():
    """Generate a single arithmetic equation string."""
    for _ in range(50):  # retry loop to enforce length constraint
        if random.random() < 0.05:
            eq = _generate_edge_case()
        else:
            op = random.choices(["+", "-", "*"], weights=[1 / 3, 1 / 3, 1 / 3])[0]

            size_a = _pick_operand_size()
            size_b = _pick_operand_size()

            a = random.randint(*_RANGES[size_a])
            b = random.randint(*_RANGES[size_b])

            if op == "+":
                eq = f"{a}+{b}={a + b}"
            elif op == "*":
                eq = f"{a}*{b}={a * b}"
            else:
                if a < b:
                    a, b = b, a
                eq = f"{a}-{b}={a - b}"

        if len(eq) <= MAX_EQ_CHARS:
            return eq

    # Fallback: guaranteed-short equation (should be extremely rare)
    a, b = random.randint(1, 99), random.randint(1, 99)
    return f"{a}+{b}={a + b}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate arithmetic datasets for MDLM training.",
    )
    parser.add_argument(
        "--num_train", type=int, default=200_000,
        help="Number of training examples (default: 200000)",
    )
    parser.add_argument(
        "--num_eval", type=int, default=10_000,
        help="Number of evaluation examples (default: 10000)",
    )
    parser.add_argument(
        "--num_test", type=int, default=2_000,
        help="Number of test examples (default: 2000)",
    )
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    total_needed = args.num_train + args.num_eval + args.num_test
    print(f"Generating {total_needed:,} unique equations...")
    unique_equations: set[str] = set()
    while len(unique_equations) < total_needed:
        unique_equations.add(generate_equation())

    unique_equations = list(unique_equations)
    random.shuffle(unique_equations)

    train_eqs = unique_equations[:args.num_train]
    eval_eqs = unique_equations[args.num_train:args.num_train + args.num_eval]
    test_eqs = unique_equations[args.num_train + args.num_eval:]

    train_file = os.path.join(args.output_dir, "arithmetic_train.jsonl")
    print(f"Writing {len(train_eqs):,} train examples -> {train_file}")
    with open(train_file, "w", encoding="utf-8") as f:
        for eq in train_eqs:
            f.write(json.dumps({"text": eq}) + "\n")

    eval_file = os.path.join(args.output_dir, "arithmetic_eval.jsonl")
    print(f"Writing {len(eval_eqs):,} eval examples -> {eval_file}")
    with open(eval_file, "w", encoding="utf-8") as f:
        for eq in eval_eqs:
            f.write(json.dumps({"text": eq}) + "\n")

    clean_test_file = os.path.join(args.output_dir, "arithmetic_test.jsonl")
    print(f"Writing {len(test_eqs):,} clean test examples -> {clean_test_file}")
    with open(clean_test_file, "w", encoding="utf-8") as f:
        for eq in test_eqs:
            f.write(json.dumps({"text": eq}) + "\n")

    print("Done!")


if __name__ == "__main__":
    main()
