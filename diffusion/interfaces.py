from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ModelOutputLike(Protocol):
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None


@runtime_checkable
class DiffusionModelLike(Protocol):
    @property
    def device(self) -> torch.device: ...

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> ModelOutputLike: ...


@runtime_checkable
class TokenizerLike(Protocol):
    mask_token_id: int | None
    eos_token_id: int | None
    bos_token_id: int | None
    unk_token_id: int | None
    pad_token_id: int | None
    padding_side: str

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str: ...

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
