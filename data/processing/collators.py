from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CollatorWrapper:
    """
    Base collator wrapper with pre/post hooks.

    Subclass and override before() and/or after() to modify inputs
    or outputs without replacing the underlying collator.
    """

    collator: Any

    def before(self, features):
        return features

    def after(self, outputs):
        return outputs

    def __call__(self, features, return_tensors=None):
        features = self.before(features)
        outputs = self.collator(features, return_tensors=return_tensors)
        outputs = self.after(outputs)
        return outputs

    def __getattr__(self, name: str):
        collator = self.__dict__.get("collator", None)
        if collator is not None:
            try:
                return getattr(collator, name)
            except AttributeError:
                pass

        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


@dataclass
class RandomTruncateWrapper(CollatorWrapper):
    """
    Collator wrapper that randomly truncates sequences to a random length.

    With probability `random_length_ratio` per batch, all sequences are logically
    truncated to a random length by zeroing out the attention mask and labels
    beyond that position. This acts as a form of length regularization.
    """

    random_length_ratio: float = 0.01
    label_pad_token_id: int = -100

    def after(self, outputs):
        if torch.rand(1) < self.random_length_ratio:
            input_ids = outputs["input_ids"]
            bsz, seq_len = input_ids.shape
            random_length = torch.randint(1, seq_len + 1, (1,), device=input_ids.device).item()

            if "attention_mask" in outputs:
                outputs["attention_mask"][:, random_length:] = 0
            else:
                attention_mask = torch.ones(
                    (bsz, seq_len),
                    dtype=torch.long,
                    device=input_ids.device,
                )
                attention_mask[:, random_length:] = 0
                outputs["attention_mask"] = attention_mask

            if "labels" in outputs:
                outputs["labels"][:, random_length:] = self.label_pad_token_id

        return outputs

@dataclass
class NoAttentionMaskWrapper(CollatorWrapper):
    """
    Wraps a data collator to remove the attention_mask from the output dictionary.
    This ensures that padded <eos_token>s remain visible to the model during Masked Diffusion training,
    which is essential for the model to correctly learn how to predict sequence termination without hallucinating.
    """

    def after(self, outputs):
        if "labels" not in outputs and "input_ids" in outputs:
            outputs["labels"] = outputs["input_ids"].clone()
        if "attention_mask" in outputs:
            del outputs["attention_mask"]
        return outputs
