"""BackPlay per-token error head."""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class BackPlayHeadConfig:
    d_model: int
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.0
    dim_feedforward: int | None = None
    hidden_state_index: int = -2
    head_type: str = "attention"


class BackPlayHead(nn.Module):
    """
    Shallow Transformer correction head for BackPlay.

    The head consumes frozen backbone hidden states and predicts per-token error
    probabilities. Higher scores mean "more likely incorrect" and are used for
    adaptive remasking during inference.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
        dim_feedforward: int | None = None,
        hidden_state_index: int = -2,
        head_type: str = "attention",
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.config = BackPlayHeadConfig(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            hidden_state_index=hidden_state_index,
            head_type=head_type,
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 1)

        if head_type == "attention":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.mlp = None
        elif head_type == "linear":
            self.encoder = None
            # Dynamically size hidden_dim to target exactly ~133k parameters
            hidden_dim = int((133000 - 3 * d_model - 2) / (d_model + 2))
            self.mlp = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        elif head_type == "scalar":
            self.encoder = None
            self.mlp = None
        else:
            raise ValueError(f"Unsupported BackPlay head type: {head_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.head_type == "attention":
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = attention_mask == 0
            h = self.encoder(hidden_states, src_key_padding_mask=key_padding_mask)
            logits = self.proj(self.norm(h)).squeeze(-1)
        elif self.config.head_type == "linear":
            h = self.norm(hidden_states)
            logits = self.mlp(h).squeeze(-1)
        else:
            h = hidden_states
            logits = self.proj(h).squeeze(-1)
        return torch.sigmoid(logits)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to_config_dict(self) -> dict:
        return asdict(self.config)

    @classmethod
    def from_config_dict(cls, config: dict) -> "BackPlayHead":
        return cls(**config)
