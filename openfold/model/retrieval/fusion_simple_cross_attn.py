from __future__ import annotations

from typing import Dict, Optional

import torch

from openfold.model.retrieval_fusion import CrossAttentionFusion

from .interfaces import FusionStrategy
from .registry import register_fusion


@register_fusion("simple_cross_attn")
class SimpleCrossAttentionFusion(FusionStrategy):
    """Thin wrapper around existing cross-attention fusion baseline."""

    def __init__(
        self,
        emb_dim: int = 1280,
        num_heads: int = 8,
        dropout: float = 0.0,
        attention_backend: str = "auto",
        flash_attn_compute_dtype: str = "bfloat16",
    ):
        super().__init__()
        self.inner = CrossAttentionFusion(
            emb_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_backend=attention_backend,
            flash_attn_compute_dtype=flash_attn_compute_dtype,
        )

    @property
    def name(self) -> str:
        return "simple_cross_attn"

    def forward(
        self,
        query_tokens: torch.Tensor,
        retrieved_tokens: torch.Tensor,
        retrieved_scores: torch.Tensor,
        retrieved_masks: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, object]] = None,
    ) -> torch.Tensor:
        del context
        return self.inner(
            query_tokens,
            retrieved_tokens,
            retrieved_scores,
            retrieved_masks=retrieved_masks,
        )
