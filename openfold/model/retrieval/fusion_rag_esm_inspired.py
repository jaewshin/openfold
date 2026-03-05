from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .cross_attention import RetrievalCrossAttention
from .interfaces import FusionStrategy
from .registry import register_fusion


@register_fusion("rag_esm_inspired")
class RagEsmInspiredFusion(FusionStrategy):
    """RAG-ESM-inspired retrieval conditioning with lightweight token updates.

    This is intentionally "inspired" rather than a strict architectural reproduction.
    """

    def __init__(
        self,
        emb_dim: int = 1280,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_hidden_mult: int = 2,
        attention_backend: str = "auto",
        flash_attn_compute_dtype: str = "bfloat16",
    ):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")
        hidden = int(emb_dim * max(1, mlp_hidden_mult))

        self.pre_norm_q = nn.LayerNorm(emb_dim)
        self.pre_norm_kv = nn.LayerNorm(emb_dim)
        self.cross_attn = RetrievalCrossAttention(
            emb_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_backend=attention_backend,
            flash_attn_compute_dtype=flash_attn_compute_dtype,
        )
        self.context_proj = nn.Linear(emb_dim, emb_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )
        self.out_norm = nn.LayerNorm(emb_dim)
        self.cross_gate = nn.Parameter(torch.tensor(0.0))
        self.mlp_gate = nn.Parameter(torch.tensor(0.0))

    @property
    def name(self) -> str:
        return "rag_esm_inspired"

    @staticmethod
    def _score_normalize(scores: torch.Tensor) -> torch.Tensor:
        if scores.numel() == 0:
            return scores
        if torch.isfinite(scores).any():
            s = torch.softmax(scores, dim=0)
            return s
        return torch.zeros_like(scores)

    @staticmethod
    def _pool_retrieved(
        retrieved_tokens: torch.Tensor,
        retrieved_masks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Pool retrieved candidates into per-candidate context vectors [K, D]."""
        if retrieved_masks is None:
            return retrieved_tokens.mean(dim=1)
        m = retrieved_masks.float().unsqueeze(-1)
        numer = (retrieved_tokens * m).sum(dim=1)
        denom = m.sum(dim=1).clamp(min=1.0)
        return numer / denom

    def forward(
        self,
        query_tokens: torch.Tensor,
        retrieved_tokens: torch.Tensor,
        retrieved_scores: torch.Tensor,
        retrieved_masks: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, object]] = None,
    ) -> torch.Tensor:
        del context
        if retrieved_tokens.shape[0] == 0:
            return query_tokens

        # [K, D]
        pooled = self._pool_retrieved(retrieved_tokens, retrieved_masks)
        scores = self._score_normalize(retrieved_scores)
        fused_context = torch.matmul(scores, pooled)  # [D]

        # Cross-attn: query tokens attend to flattened retrieved tokens.
        q = self.pre_norm_q(query_tokens)  # [Nq, D]
        kv = self.pre_norm_kv(retrieved_tokens.reshape(-1, retrieved_tokens.shape[-1]))
        flat_mask = retrieved_masks.reshape(-1).bool() if retrieved_masks is not None else None
        attn_out = self.cross_attn(
            query_tokens=q,
            key_value_tokens=kv,
            kv_valid_mask=flat_mask,
        )

        context_bias = self.context_proj(fused_context).unsqueeze(0).expand_as(query_tokens)
        x = query_tokens + torch.sigmoid(self.cross_gate) * (attn_out + context_bias)
        x = x + torch.sigmoid(self.mlp_gate) * self.mlp(x)
        return self.out_norm(x)
