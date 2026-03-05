from __future__ import annotations

import importlib.util
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(
        f"Unsupported dtype={name!r}. Expected one of "
        "{'bfloat16', 'float16', 'float32'}."
    )


class RetrievalCrossAttention(nn.Module):
    """Cross-attention block with optional flash-attn backend."""

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_backend: str = "auto",
        flash_attn_compute_dtype: str = "bfloat16",
    ):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")

        backend = str(attention_backend).strip().lower()
        if backend not in {"auto", "torch", "sdpa", "flash_attn"}:
            raise ValueError(
                f"Unsupported attention_backend={attention_backend!r}. "
                "Expected one of {'auto', 'torch', 'sdpa', 'flash_attn'}."
            )

        self.emb_dim = int(emb_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(self.emb_dim // self.num_heads)
        self.dropout = float(dropout)
        self.attention_backend = backend
        self.flash_attn_compute_dtype = _parse_dtype(flash_attn_compute_dtype)

        self.q_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.k_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.v_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim)

    @staticmethod
    def _has_sdpa() -> bool:
        return hasattr(F, "scaled_dot_product_attention")

    @staticmethod
    def _has_flash_attn() -> bool:
        return importlib.util.find_spec("flash_attn") is not None

    def _resolve_backend(self, query_tokens: torch.Tensor) -> str:
        if self.attention_backend == "torch":
            return "torch"
        if self.attention_backend == "sdpa":
            return "sdpa" if self._has_sdpa() else "torch"
        if self.attention_backend == "flash_attn":
            if query_tokens.is_cuda and self._has_flash_attn():
                return "flash_attn"
            return "sdpa" if self._has_sdpa() else "torch"

        # auto
        if query_tokens.is_cuda and self._has_flash_attn():
            return "flash_attn"
        if self._has_sdpa():
            return "sdpa"
        return "torch"

    def _project_qkv(
        self,
        query_tokens: torch.Tensor,
        key_value_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(query_tokens)  # [Nq, D]
        k = self.k_proj(key_value_tokens)  # [Nk, D]
        v = self.v_proj(key_value_tokens)  # [Nk, D]
        return q, k, v

    def _forward_torch(
        self,
        query_tokens: torch.Tensor,
        key_value_tokens: torch.Tensor,
    ) -> torch.Tensor:
        n_q = int(query_tokens.shape[0])
        n_kv = int(key_value_tokens.shape[0])
        if n_kv == 0:
            return torch.zeros_like(query_tokens)

        q, k, v = self._project_qkv(query_tokens, key_value_tokens)
        q = q.view(n_q, self.num_heads, self.head_dim).permute(1, 0, 2)  # [H, Nq, d]
        k = k.view(n_kv, self.num_heads, self.head_dim).permute(1, 0, 2)  # [H, Nk, d]
        v = v.view(n_kv, self.num_heads, self.head_dim).permute(1, 0, 2)  # [H, Nk, d]

        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(self.head_dim))
        attn = torch.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)  # [H, Nq, d]
        out = out.permute(1, 0, 2).contiguous().view(n_q, self.emb_dim)
        out = self.out_proj(out)
        return out.to(dtype=query_tokens.dtype)

    def _forward_sdpa(
        self,
        query_tokens: torch.Tensor,
        key_value_tokens: torch.Tensor,
    ) -> torch.Tensor:
        n_q = int(query_tokens.shape[0])
        n_kv = int(key_value_tokens.shape[0])
        if n_kv == 0:
            return torch.zeros_like(query_tokens)

        q, k, v = self._project_qkv(query_tokens, key_value_tokens)
        q = q.view(n_q, self.num_heads, self.head_dim).permute(1, 0, 2).unsqueeze(0)  # [1, H, Nq, d]
        k = k.view(n_kv, self.num_heads, self.head_dim).permute(1, 0, 2).unsqueeze(0)  # [1, H, Nk, d]
        v = v.view(n_kv, self.num_heads, self.head_dim).permute(1, 0, 2).unsqueeze(0)  # [1, H, Nk, d]

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=(self.dropout if self.training else 0.0),
            is_causal=False,
        )  # [1, H, Nq, d]
        out = out.squeeze(0).permute(1, 0, 2).contiguous().view(n_q, self.emb_dim)
        out = self.out_proj(out)
        return out.to(dtype=query_tokens.dtype)

    def _forward_flash_attn(
        self,
        query_tokens: torch.Tensor,
        key_value_tokens: torch.Tensor,
    ) -> torch.Tensor:
        n_q = int(query_tokens.shape[0])
        n_kv = int(key_value_tokens.shape[0])
        if n_kv == 0:
            return torch.zeros_like(query_tokens)

        from flash_attn import flash_attn_func

        q, k, v = self._project_qkv(query_tokens, key_value_tokens)
        q = q.view(1, n_q, self.num_heads, self.head_dim)
        k = k.view(1, n_kv, self.num_heads, self.head_dim)
        v = v.view(1, n_kv, self.num_heads, self.head_dim)

        target_dtype = self.flash_attn_compute_dtype
        q = q.to(dtype=target_dtype)
        k = k.to(dtype=target_dtype)
        v = v.to(dtype=target_dtype)

        out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=(self.dropout if self.training else 0.0),
            causal=False,
        )  # [1, Nq, H, d]

        out = out.reshape(n_q, self.emb_dim).to(dtype=self.out_proj.weight.dtype)
        out = self.out_proj(out)
        return out.to(dtype=query_tokens.dtype)

    def forward(
        self,
        query_tokens: torch.Tensor,
        key_value_tokens: torch.Tensor,
        kv_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query_tokens.ndim != 2:
            raise ValueError(f"query_tokens must be rank-2 [N, D], got shape={tuple(query_tokens.shape)}")
        if key_value_tokens.ndim != 2:
            raise ValueError(
                f"key_value_tokens must be rank-2 [N, D], got shape={tuple(key_value_tokens.shape)}"
            )

        if kv_valid_mask is not None:
            valid = kv_valid_mask.bool()
            if valid.numel() != key_value_tokens.shape[0]:
                raise ValueError(
                    "kv_valid_mask length must match key_value_tokens length. "
                    f"Got {valid.numel()} vs {key_value_tokens.shape[0]}."
                )
            if not bool(valid.any().item()):
                return torch.zeros_like(query_tokens)
            key_value_tokens = key_value_tokens[valid]

        backend = self._resolve_backend(query_tokens)
        if backend == "flash_attn":
            try:
                return self._forward_flash_attn(query_tokens, key_value_tokens)
            except Exception:
                # Keep runs robust if flash-attn is unavailable for a given build/dtype.
                if self._has_sdpa():
                    return self._forward_sdpa(query_tokens, key_value_tokens)
                return self._forward_torch(query_tokens, key_value_tokens)
        if backend == "sdpa":
            return self._forward_sdpa(query_tokens, key_value_tokens)
        return self._forward_torch(query_tokens, key_value_tokens)
