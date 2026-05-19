from __future__ import annotations

import json
import os
from functools import partial

from datasets import load_dataset

from data.processing.tokenization import tokenize_and_pad
from .base import TaskAdapter


class ArithmeticTaskAdapter(TaskAdapter):
    def __init__(self):
        super().__init__(
            name="arithmetic",
            default_data_path="data/arithmetic_train.jsonl",
            default_inference_dataset_path="data/arithmetic_eval.jsonl",
            default_tokenizer_path="./tokenizers/arithmetic_char_tokenizer",
            default_prompt_delimiter="=",
            default_prompts=["451+301="],
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
        """Load arithmetic JSONL files and return (train_dataset, eval_dataset)."""
        split = f"train[:{limit_data}]" if limit_data and limit_data > 0 else "train"

        tokenize = partial(
            tokenize_and_pad,
            tokenizer=tokenizer,
            text_field="text",
            seq_length=seq_len,
            insert_eos=True,
            mask_until_token=mask_until_token,
        )

        raw_train = load_dataset("json", data_files=data_path, split=split)
        train_dataset = raw_train.map(tokenize, batched=True, remove_columns=raw_train.column_names)
        print(f"  {len(train_dataset):,} padded training sequences of length {seq_len}")

        eval_dataset = None
        if eval_data_path and os.path.exists(eval_data_path):
            raw_eval = load_dataset("json", data_files=eval_data_path, split=split)
            eval_dataset = raw_eval.map(tokenize, batched=True, remove_columns=raw_eval.column_names)
            print(f"  {len(eval_dataset):,} padded evaluation sequences.")

        return train_dataset, eval_dataset

    # ------------------------------------------------------------------
    # Inspection / inference helpers
    # ------------------------------------------------------------------

    def load_examples(self, data_path: str, offset: int, limit: int) -> list[str]:
        examples: list[str] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx < offset:
                    continue
                if len(examples) >= limit:
                    break
                record = json.loads(line)
                examples.append(record["text"])
        return examples

    def load_dataset_prompts(
        self,
        path: str,
        mode: str,
        num: int,
        delimiter: str | None = None,
    ) -> list[str]:
        if not os.path.exists(path):
            return []

        with open(path, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict) and data.get("text") is not None:
                    lines.append(data)

        if not lines:
            return []

        if num <= 0:
            num = len(lines)

        if mode == "random":
            import random
            selected = random.sample(lines, min(num, len(lines)))
        else:
            selected = lines[:num]

        prompt_delimiter = delimiter or self.default_prompt_delimiter
        self._solution_strings = [
            item.get("ground_truth") or item.get("text", "")
            for item in selected
        ]
        if prompt_delimiter is None:
            return [item["text"] for item in selected]

        prompts = []
        for item in selected:
            text = item.get("text", "")
            # Corrupted datasets include ground_truth; keep full equation so
            # inference sees the mistaken RHS and can correct it via infill.
            if "ground_truth" in item:
                prompts.append(text)
                continue
            if prompt_delimiter in text:
                prompts.append(text.split(prompt_delimiter)[0] + prompt_delimiter)
            else:
                prompts.append(text)
        return prompts

    def describe_example(self, text: str) -> list[tuple[str, str]]:
        eq_pos = text.rfind("=")
        if eq_pos >= 0:
            return [("'=' position", str(eq_pos))]
        return []
