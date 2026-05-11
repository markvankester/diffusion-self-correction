"""
Generate corrupted arithmetic test data from an existing clean test set.
"""

import argparse
import json
import random


def inject_error(
    equation: str,
    min_errors: int = 1,
    max_errors: int | None = None,
) -> str:
    """
    Inject one or more digit errors into the equation RHS.

    Example:
      123+45=168 -> 123+45=268
    """
    lhs, rhs = equation.split("=", 1)
    if not rhs:
        return equation

    upper = len(rhs) if max_errors is None else max_errors
    min_errors = max(1, min(min_errors, len(rhs)))
    upper = max(min_errors, min(upper, len(rhs)))
    n_errors = random.randint(min_errors, upper)

    positions = random.sample(range(len(rhs)), k=n_errors)
    rhs_chars = list(rhs)
    for pos in positions:
        orig = int(rhs_chars[pos])
        rhs_chars[pos] = str(random.choice([d for d in range(10) if d != orig]))
    return f"{lhs}={''.join(rhs_chars)}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate a corrupted test set by injecting RHS digit errors.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/arithmetic_test.jsonl",
        help="Path to clean test JSONL (default: data/arithmetic_test.jsonl)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/arithmetic_test_corrupted.jsonl",
        help="Path to write corrupted JSONL (default: data/arithmetic_test_corrupted.jsonl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--min_errors",
        type=int,
        default=1,
        help="Minimum RHS digit errors per equation (default: 1)",
    )
    parser.add_argument(
        "--max_errors",
        type=int,
        default=None,
        help="Maximum RHS digit errors per equation (default: all RHS digits)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    equations: list[str] = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            equations.append(record["text"])

    with open(args.output_file, "w", encoding="utf-8") as f:
        for eq in equations:
            corrupted = inject_error(
                eq,
                min_errors=args.min_errors,
                max_errors=args.max_errors,
            )
            f.write(json.dumps({"text": corrupted, "ground_truth": eq}) + "\n")

    print(f"Wrote {len(equations):,} corrupted examples to: {args.output_file}")


if __name__ == "__main__":
    main()
