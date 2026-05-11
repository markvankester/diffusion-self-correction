from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskAdapter:
    name: str
    default_data_path: str
    default_inference_dataset_path: str
    default_tokenizer_path: str
    default_prompt_delimiter: str | None
    default_prompts: list[str]

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
        Build train and (optionally) eval HuggingFace Datasets.

        Subclasses must implement this method.  The returned tuple is
        (train_dataset, eval_dataset), where eval_dataset may be None.

        Args:
            tokenizer:        Loaded tokenizer.
            data_path:        Path to the primary training data.
            seq_len:          Token sequence length used during training.
            eval_data_path:   Optional separate eval data file (used by
                              arithmetic; ignored by sudoku which uses
                              eval_fraction instead).
            limit_data:       Maximum rows to load (0 = all).
            mask_until_token: Token string up to which labels are set to -100
                              (arithmetic-style LHS masking; ignored by sudoku
                              which uses per-cell clue masks).
            eval_fraction:    Fraction of training data to reserve as eval
                              when no eval_data_path is provided (sudoku).
        """
        raise NotImplementedError

    def load_examples(self, data_path: str, offset: int, limit: int) -> list[str]:
        raise NotImplementedError

    def load_dataset_prompts(
        self,
        path: str,
        mode: str,
        num: int,
        delimiter: str | None = None,
    ) -> list[str]:
        raise NotImplementedError

    def describe_example(self, text: str) -> list[tuple[str, str]]:
        return []


def get_task_adapter(task: str) -> TaskAdapter:
    if task == "arithmetic":
        from .arithmetic import ArithmeticTaskAdapter

        return ArithmeticTaskAdapter()
    if task == "sudoku":
        from .sudoku import SudokuTaskAdapter

        return SudokuTaskAdapter()
    raise ValueError(f"Unsupported task: {task}")
