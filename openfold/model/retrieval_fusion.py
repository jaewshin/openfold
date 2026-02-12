"""
Retrieval-augmented fusion modules for OpenFold SoloSeq.

Architecture:
    query_emb [N, D]  (ESM-1b)
           |
     EmbeddingRetriever  →  top-K retrieved embeddings [K, N', D]
           |
     CrossAttentionFusion  →  fused_emb [N, D]
           |
     (frozen) PreembeddingEmbedder → Evoformer → StructureModule → Loss
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingRetriever(nn.Module):
    """Differentiable retriever over a database of precomputed embeddings.

    Scores each database entry against the query using a learned projection
    and returns the top-K entries with their (softmax-normalised) scores.

    The scoring function is:
        score(q, d) = (W_q @ mean_pool(q))^T  (W_k @ mean_pool(d))  / sqrt(c_proj)

    where W_q, W_k are learned projections.  This keeps the retrieval
    differentiable w.r.t. the query embedding so that the OpenFold loss
    can back-propagate into the retriever.

    Args:
        emb_dim:   Dimension of the input embeddings (e.g. 1280 for ESM-1b).
        c_proj:    Dimension of the query/key projection space.
        top_k:     Number of entries to retrieve.
    """

    def __init__(
        self,
        emb_dim: int = 1280,
        c_proj: int = 128,
        top_k: int = 16,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.c_proj = c_proj
        self.top_k = top_k

        self.query_proj = nn.Linear(emb_dim, c_proj, bias=False)
        self.key_proj = nn.Linear(emb_dim, c_proj, bias=False)

    def _pool(self, emb: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean-pool over the residue dimension.

        Args:
            emb:  [*, N, D]  per-residue embedding.
            mask: [*, N]     residue mask (1 = valid).  If None, all valid.

        Returns:
            [*, D] pooled vector.
        """
        if mask is None:
            return emb.mean(dim=-2)
        mask = mask.unsqueeze(-1)  # [*, N, 1]
        return (emb * mask).sum(dim=-2) / mask.sum(dim=-2).clamp(min=1)

    def forward(
        self,
        query_emb: torch.Tensor,
        db_embs: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        db_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve the top-K database entries most similar to the query.

        Args:
            query_emb:  [N_q, D]      per-residue query embedding.
            db_embs:    [M, N_d, D]   database of M embeddings, each of
                                       length N_d (may be padded).
            query_mask: [N_q]         optional residue mask for the query.
            db_masks:   [M, N_d]      optional residue masks for database
                                       entries.

        Returns:
            scores:     [K]           softmax-normalised retrieval scores.
            indices:    [K]           indices into the database (long).
            retrieved:  [K, N_d, D]   the retrieved embeddings.
        """
        # Pool to sequence-level representations
        q = self._pool(query_emb, query_mask)          # [D]
        d = self._pool(db_embs, db_masks)              # [M, D]

        # Project
        q_proj = self.query_proj(q)                     # [c_proj]
        k_proj = self.key_proj(d)                       # [M, c_proj]

        # Scaled dot-product similarity
        logits = torch.matmul(k_proj, q_proj) / (self.c_proj ** 0.5)  # [M]

        # Top-K selection (straight-through for differentiability)
        k = min(self.top_k, logits.shape[0])
        topk_logits, topk_indices = logits.topk(k)     # [K], [K]

        # Softmax over selected entries only
        scores = F.softmax(topk_logits, dim=0)          # [K]

        # Gather retrieved embeddings
        retrieved = db_embs[topk_indices]                # [K, N_d, D]

        return scores, topk_indices, retrieved


class CrossAttentionFusion(nn.Module):
    """Fuse retrieved embeddings into the query embedding via cross-attention.

    The query attends to each retrieved embedding (per-residue) using
    multi-head cross-attention, then the K attention outputs are combined
    using the retrieval scores as mixture weights:

        fused = query + sum_k( score_k * CrossAttn(query, retrieved_k) )

    Because retrieved sequences may differ in length from the query, a
    simple position-agnostic cross-attention is used (the positional
    information is already baked into the ESM-1b embeddings).

    Args:
        emb_dim:     Embedding dimension (1280 for ESM-1b).
        num_heads:   Number of attention heads.
        dropout:     Dropout probability on attention weights.
    """

    def __init__(
        self,
        emb_dim: int = 1280,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        assert emb_dim % num_heads == 0

        self.head_dim = emb_dim // num_heads

        # Query comes from the input embedding, K/V from retrieved
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

        self.layer_norm_q = nn.LayerNorm(emb_dim)
        self.layer_norm_kv = nn.LayerNorm(emb_dim)
        self.layer_norm_out = nn.LayerNorm(emb_dim)

        self.dropout = nn.Dropout(dropout)

        # Gate so the module can learn to be a no-op initially
        self.gate = nn.Parameter(torch.zeros(1))

    def _cross_attn(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single cross-attention pass.

        Args:
            query:     [N_q, D]
            key_value: [N_kv, D]
            kv_mask:   [N_kv]  optional mask (1 = valid).

        Returns:
            [N_q, D] cross-attention output.
        """
        N_q = query.shape[0]
        N_kv = key_value.shape[0]
        H = self.num_heads
        d = self.head_dim

        q = self.q_proj(self.layer_norm_q(query))         # [N_q, D]
        k = self.k_proj(self.layer_norm_kv(key_value))    # [N_kv, D]
        v = self.v_proj(self.layer_norm_kv(key_value))    # [N_kv, D]

        # Reshape to multi-head: [H, N, d]
        q = q.view(N_q, H, d).permute(1, 0, 2)           # [H, N_q, d]
        k = k.view(N_kv, H, d).permute(1, 0, 2)          # [H, N_kv, d]
        v = v.view(N_kv, H, d).permute(1, 0, 2)          # [H, N_kv, d]

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)  # [H, N_q, N_kv]

        if kv_mask is not None:
            attn = attn.masked_fill(
                ~kv_mask.bool().unsqueeze(0).unsqueeze(1),  # [1, 1, N_kv]
                float("-inf"),
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                       # [H, N_q, d]
        out = out.permute(1, 0, 2).contiguous().view(N_q, -1)  # [N_q, D]
        out = self.out_proj(out)
        return out

    def forward(
        self,
        query_emb: torch.Tensor,
        retrieved_embs: torch.Tensor,
        scores: torch.Tensor,
        retrieved_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse retrieved embeddings into the query via scored cross-attention.

        Args:
            query_emb:       [N_q, D]     per-residue query embedding.
            retrieved_embs:  [K, N_kv, D] retrieved database embeddings.
            scores:          [K]          retrieval scores (sum to 1).
            retrieved_masks: [K, N_kv]    optional masks for retrieved seqs.

        Returns:
            fused_emb: [N_q, D]  the fused embedding (same shape as input).
        """
        K = retrieved_embs.shape[0]
        fused = torch.zeros_like(query_emb)  # [N_q, D]

        for i in range(K):
            kv_mask = retrieved_masks[i] if retrieved_masks is not None else None
            attn_out = self._cross_attn(query_emb, retrieved_embs[i], kv_mask)
            fused = fused + scores[i] * attn_out

        # Gated residual
        fused = self.layer_norm_out(query_emb + torch.sigmoid(self.gate) * fused)

        return fused
