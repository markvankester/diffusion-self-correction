"""
PRISM Per-Token Quality Head
=============================
Lightweight adapter that attaches to a pretrained MDM backbone to predict
per-token quality scores g_θ(y) ∈ [0, 1]^L.

For each non-masked position i, the score approximates:
    g_θ^i(y) ≈ p(x_i = y_i | y ⊕ m_i)

i.e., how likely the token at position i would be if that position were masked
and predicted from the rest of the sequence.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class PRISMHeadConfig:
    d_model: int
    head_type: str = "attention"
    n_heads: int = 4
    dropout: float = 0.0


class PRISMHead(nn.Module):
    """
    Per-token quality scoring head for PRISM self-correction.

    Takes the final hidden state h ∈ R^{B×L×D} from the MDM backbone
    and produces quality scores q ∈ [0, 1]^{B×L}.

    Args:
        d_model: Hidden dimension of the backbone (must match).
        n_heads: Number of attention heads in the adapter layer.
        dropout: Dropout rate for the attention layer.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int | None = 4,
        dropout: float | None = 0.0,
        head_type: str = "attention",
    ):
        super().__init__()
        if n_heads is None:
            n_heads = 4
        if dropout is None:
            dropout = 0.0
        self.config = PRISMHeadConfig(
            d_model=d_model,
            head_type=head_type,
            n_heads=n_heads,
            dropout=dropout,
        )

        if head_type == "attention":
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.layer_norm = nn.LayerNorm(d_model)
            self.mlp = None
        elif head_type == "linear":
            self.attn = None
            self.layer_norm = nn.LayerNorm(d_model)
            hidden_dim = 2 * d_model
            self.mlp = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        elif head_type == "scalar":
            self.attn = None
            self.layer_norm = None
            self.mlp = None
        else:
            raise ValueError(f"Unsupported PRISM head type: {head_type}")

        self.proj = nn.Linear(d_model, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, D] final hidden state from the MDM backbone.
            attention_mask: Optional [B, L] mask where 1 marks valid tokens and 0 marks padding.

        Returns:
            quality_scores: [B, L] per-token quality scores in [0, 1].
        """
        if self.config.head_type == "attention":
            residual = hidden_states
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = attention_mask == 0

            attn_out, _ = self.attn(
                hidden_states,
                hidden_states,
                hidden_states,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            h = self.layer_norm(residual + attn_out)
            logits = self.proj(h).squeeze(-1)  # [B, L]
        elif self.config.head_type == "linear":
            h = self.layer_norm(hidden_states)
            logits = self.mlp(h).squeeze(-1)  # [B, L]
        else:
            h = hidden_states
            logits = self.proj(h).squeeze(-1)  # [B, L]
        return torch.sigmoid(logits)

    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to_config_dict(self) -> dict:
        return asdict(self.config)

    @classmethod
    def from_config_dict(cls, config: dict) -> "PRISMHead":
        return cls(**config)
