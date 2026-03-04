from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .interfaces import QueryPipeline, QueryVectors


class LegacyQueryPipeline(QueryPipeline):
    """Legacy retrieval query pipeline: pooled seq_embedding + linear projections."""

    def __init__(
        self,
        seq_embedding_dim: int = 1280,
        seq_index_dim: int = 1280,
        struct_index_dim: int = 512,
    ):
        super().__init__()
        self.seq_query_proj = nn.Linear(seq_embedding_dim, seq_index_dim)
        self.struct_query_proj = nn.Linear(seq_embedding_dim, struct_index_dim)

    @staticmethod
    def _pool_query(tokens: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # tokens: [B, N, D], mask: [B, N]
        if mask is None:
            return tokens.mean(dim=1)
        m = mask.float().unsqueeze(-1)
        return (tokens * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def encode_queries(
        self,
        query_tokens: torch.Tensor,
        query_mask: Optional[torch.Tensor],
        raw_sequences: Optional[Sequence[str]],
        use_seq: bool,
        use_struct: bool,
    ) -> QueryVectors:
        del raw_sequences
        pooled = self._pool_query(query_tokens, query_mask)

        seq = self.seq_query_proj(pooled) if use_seq else None
        struct = self.struct_query_proj(pooled) if use_struct else None
        return QueryVectors(seq=seq, struct=struct)
