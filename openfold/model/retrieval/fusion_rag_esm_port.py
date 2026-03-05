from __future__ import annotations

from typing import Dict, Optional, Sequence, Set

import torch
import torch.nn as nn

from .cross_attention import RetrievalCrossAttention
from .interfaces import FusionStrategy
from .registry import register_fusion


class _RagFusionBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        mlp_hidden_mult: int,
        attention_backend: str = "auto",
        flash_attn_compute_dtype: str = "bfloat16",
    ):
        super().__init__()
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
        self.ff_norm = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )
        self.cross_gate = nn.Parameter(torch.tensor(0.0))
        self.ff_gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        retrieved_tokens: torch.Tensor,
        scores: torch.Tensor,
        retrieved_masks: Optional[torch.Tensor],
        use_cross_attention: bool,
    ):
        cross = torch.zeros_like(x)
        valid_hits = 0
        context_tokens = 0

        if use_cross_attention and retrieved_tokens.shape[0] > 0:
            q = self.pre_norm_q(x)
            for k in range(retrieved_tokens.shape[0]):
                score = scores[k]
                if not torch.isfinite(score) or float(score.item()) <= 0.0:
                    continue

                kv = self.pre_norm_kv(retrieved_tokens[k].to(dtype=x.dtype))
                valid_mask_k = None
                if retrieved_masks is not None:
                    mask_k = retrieved_masks[k].bool()
                    if not mask_k.any():
                        continue
                    valid_mask_k = mask_k
                    context_tokens += int(mask_k.sum().item())
                else:
                    context_tokens += int(retrieved_tokens.shape[1])

                attn_out = self.cross_attn(
                    query_tokens=q,
                    key_value_tokens=kv,
                    kv_valid_mask=valid_mask_k,
                )
                cross = cross + score.to(dtype=x.dtype) * attn_out
                valid_hits += 1

        x = x + torch.sigmoid(self.cross_gate) * cross
        x = x + torch.sigmoid(self.ff_gate) * self.ffn(self.ff_norm(x))
        return x, valid_hits, context_tokens


@register_fusion("rag_esm_port")
class RagEsmPortFusion(FusionStrategy):
    """RAG-ESM-style stacked gated cross-attention + FFN token updater."""

    def __init__(
        self,
        emb_dim: int = 1280,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_hidden_mult: int = 2,
        num_blocks: int = 4,
        layers_with_cross_attention: object = "all",
        skip_cross_ratio: float = 0.0,
        attention_backend: str = "auto",
        flash_attn_compute_dtype: str = "bfloat16",
    ):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")
        if int(num_blocks) <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        if not (0.0 <= float(skip_cross_ratio) < 1.0):
            raise ValueError(f"skip_cross_ratio must be in [0, 1), got {skip_cross_ratio}")

        self.num_blocks = int(num_blocks)
        self.blocks = nn.ModuleList(
            [
                _RagFusionBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    mlp_hidden_mult=mlp_hidden_mult,
                    attention_backend=attention_backend,
                    flash_attn_compute_dtype=flash_attn_compute_dtype,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.cross_attention_layers = self._resolve_cross_layers(
            layers_with_cross_attention=layers_with_cross_attention,
            num_blocks=self.num_blocks,
        )
        self.skip_cross_ratio = float(skip_cross_ratio)
        self.out_norm = nn.LayerNorm(emb_dim)
        self._last_stats: Dict[str, float] = {
            "skip_cross_applied": 0.0,
            "retrieval_valid_hits": 0.0,
            "context_tokens_used": 0.0,
        }

    @property
    def name(self) -> str:
        return "rag_esm_port"

    @staticmethod
    def _score_normalize(scores: torch.Tensor) -> torch.Tensor:
        if scores.numel() == 0:
            return scores
        finite = torch.isfinite(scores)
        if finite.any():
            masked = scores.masked_fill(~finite, -1e9)
            return torch.softmax(masked, dim=0)
        return torch.zeros_like(scores)

    @staticmethod
    def _resolve_cross_layers(layers_with_cross_attention: object, num_blocks: int) -> Set[int]:
        if isinstance(layers_with_cross_attention, str):
            key = layers_with_cross_attention.strip().lower()
            if key == "all":
                return set(range(num_blocks))
            if key == "last":
                return {num_blocks - 1}
            if key == "none":
                return set()
            if key.startswith("alternate"):
                step = 2
                parts = key.split("-", 1)
                if len(parts) == 2 and parts[1]:
                    step = max(1, int(parts[1]))
                return set(range(num_blocks - 1, -1, -step))
            raise ValueError(
                f"Unsupported layers_with_cross_attention={layers_with_cross_attention!r}. "
                "Expected one of {'all', 'last', 'none', 'alternate-N'} or a list of indices."
            )

        if isinstance(layers_with_cross_attention, Sequence):
            out = {int(v) for v in layers_with_cross_attention}
            invalid = sorted(v for v in out if v < 0 or v >= num_blocks)
            if invalid:
                raise ValueError(
                    f"layers_with_cross_attention contains invalid block indices: {invalid} "
                    f"for num_blocks={num_blocks}"
                )
            return out

        raise ValueError(
            "layers_with_cross_attention must be a string policy or a sequence of block indices."
        )

    def get_last_stats(self) -> Dict[str, float]:
        return dict(self._last_stats)

    def forward(
        self,
        query_tokens: torch.Tensor,
        retrieved_tokens: torch.Tensor,
        retrieved_scores: torch.Tensor,
        retrieved_masks: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, object]] = None,
    ) -> torch.Tensor:
        if context is not None:
            if "retrieved_tokens" in context:
                retrieved_tokens = context["retrieved_tokens"]  # type: ignore[assignment]
            if retrieved_masks is None and "retrieved_masks" in context:
                retrieved_masks = context["retrieved_masks"]  # type: ignore[assignment]
            if "retrieved_scores" in context:
                retrieved_scores = context["retrieved_scores"]  # type: ignore[assignment]

        if retrieved_tokens.shape[0] == 0:
            self._last_stats = {
                "skip_cross_applied": 0.0,
                "retrieval_valid_hits": 0.0,
                "context_tokens_used": 0.0,
            }
            return query_tokens

        skip_cross = False
        if self.training and self.skip_cross_ratio > 0.0:
            skip_cross = bool((torch.rand((), device=query_tokens.device) < self.skip_cross_ratio).item())

        scores = self._score_normalize(retrieved_scores)
        x = query_tokens
        block_valid_hits = 0
        block_context_tokens = 0

        for i, block in enumerate(self.blocks):
            x, valid_hits, context_tokens = block(
                x=x,
                retrieved_tokens=retrieved_tokens,
                scores=scores,
                retrieved_masks=retrieved_masks,
                use_cross_attention=((i in self.cross_attention_layers) and (not skip_cross)),
            )
            block_valid_hits = max(block_valid_hits, int(valid_hits))
            block_context_tokens = max(block_context_tokens, int(context_tokens))

        self._last_stats = {
            "skip_cross_applied": float(skip_cross),
            "retrieval_valid_hits": float(block_valid_hits),
            "context_tokens_used": float(block_context_tokens),
        }
        return self.out_norm(x)
