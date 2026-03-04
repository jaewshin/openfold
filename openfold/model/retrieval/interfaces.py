from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn


@dataclass
class QueryVectors:
    """Per-source query vectors for retrieval search."""

    seq: Optional[torch.Tensor] = None
    struct: Optional[torch.Tensor] = None


class FusionStrategy(nn.Module, ABC):
    """Common interface for retrieval fusion strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        query_tokens: torch.Tensor,
        retrieved_tokens: torch.Tensor,
        retrieved_scores: torch.Tensor,
        retrieved_masks: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, object]] = None,
    ) -> torch.Tensor:
        """Fuse retrieved tokens into query token embeddings.

        Args:
            query_tokens: [Nq, D]
            retrieved_tokens: [K, Nk, D]
            retrieved_scores: [K]
            retrieved_masks: [K, Nk] optional 0/1 mask
            context: optional source metadata

        Returns:
            Fused query embeddings [Nq, D].
        """
        raise NotImplementedError


class QueryPipeline(nn.Module, ABC):
    """Encodes batches into retrieval query vectors for each source."""

    @abstractmethod
    def encode_queries(
        self,
        query_tokens: torch.Tensor,
        query_mask: Optional[torch.Tensor],
        raw_sequences: Optional[Sequence[str]],
        use_seq: bool,
        use_struct: bool,
    ) -> QueryVectors:
        """Encode query vectors for enabled retrieval sources.

        Args:
            query_tokens: [B, N, D]
            query_mask: [B, N] optional
            raw_sequences: list[str] optional metadata
            use_seq: whether sequence source is enabled
            use_struct: whether structure source is enabled
        """
        raise NotImplementedError


class InjectionStrategy(nn.Module, ABC):
    """Injects fused retrieval tokens into OpenFold input feature tensors."""

    @abstractmethod
    def apply(
        self,
        tensor_batch: Dict[str, torch.Tensor],
        fused_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
