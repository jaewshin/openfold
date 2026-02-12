"""
Retrieval-Augmented Folding model built on top of frozen OpenFold SoloSeq.

The idea: given a query protein sequence, use its ESM-1b embedding to
retrieve similar embeddings from a database, fuse them via cross-attention,
and feed the fused embedding into a frozen OpenFold-SoloSeq model.  The
structural loss from OpenFold drives the training of the retriever and
the fusion module while all OpenFold parameters stay frozen.

Usage:
    See ``train_retrieval_fusion.py`` for the full training loop.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.model.retrieval_fusion import (
    CrossAttentionFusion,
    EmbeddingRetriever,
)
from openfold.utils.loss import AlphaFoldLoss
from openfold.utils.tensor_utils import tensor_tree_map

logger = logging.getLogger(__name__)


class RetrievalAugmentedFolding(nn.Module):
    """Wraps a frozen OpenFold-SoloSeq with a trainable retriever + fusion head.

    Forward flow:
        1. ``EmbeddingRetriever`` scores the query against a database and
           returns top-K embeddings with differentiable scores.
        2. ``CrossAttentionFusion`` fuses the retrieved embeddings into the
           query embedding via gated, score-weighted cross-attention.
        3. The fused embedding replaces ``feats["seq_embedding"]`` in the
           batch and is passed through the frozen ``AlphaFold`` model.

    Because the AlphaFold parameters have ``requires_grad=False`` but the
    forward pass still builds a computation graph, gradients from the
    structural loss flow back through the frozen activations into the
    retriever and fusion parameters.

    Args:
        config:          Full OpenFold config (should be a seqemb preset).
        emb_dim:         Dimension of pre-computed embeddings (1280 for ESM-1b).
        retriever_proj:  Projection dimension for the retriever.
        top_k:           Number of database entries to retrieve.
        fusion_heads:    Number of attention heads in the fusion module.
        fusion_dropout:  Dropout in the fusion cross-attention.
    """

    def __init__(
        self,
        config,
        emb_dim: int = 1280,
        retriever_proj: int = 128,
        top_k: int = 16,
        fusion_heads: int = 8,
        fusion_dropout: float = 0.0,
    ):
        super().__init__()

        # --- Frozen backbone --------------------------------------------------
        self.openfold = AlphaFold(config)
        self.openfold.eval()
        for p in self.openfold.parameters():
            p.requires_grad_(False)

        # --- Trainable modules ------------------------------------------------
        self.retriever = EmbeddingRetriever(
            emb_dim=emb_dim,
            c_proj=retriever_proj,
            top_k=top_k,
        )
        self.fusion = CrossAttentionFusion(
            emb_dim=emb_dim,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Yield only the parameters that should be optimised."""
        yield from self.retriever.parameters()
        yield from self.fusion.parameters()

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: dict,
        db_embs: torch.Tensor,
        db_masks: Optional[torch.Tensor] = None,
    ) -> dict:
        """Run retrieval-augmented folding.

        Args:
            batch:    Standard OpenFold feature dict (with recycling dim).
                      Must contain ``"seq_embedding"`` (SoloSeq mode).
            db_embs:  [M, N_d, D]  pre-loaded embedding database.
            db_masks: [M, N_d]     optional residue masks for db entries.

        Returns:
            outputs:  OpenFold output dict (same as ``AlphaFold.forward``).
                      Also includes ``"retrieval_scores"`` and
                      ``"retrieval_indices"`` for analysis.
        """
        # The query embedding is stored with a recycling dim at the end.
        # We use the first recycling copy (they are all the same for
        # seq_embedding since it doesn't change across recycles).
        query_emb = batch["seq_embedding"][..., 0]  # [N_q, D]
        query_mask = batch["seq_mask"][..., 0] if "seq_mask" in batch else None

        # 1. Retrieve
        scores, indices, retrieved = self.retriever(
            query_emb, db_embs,
            query_mask=query_mask,
            db_masks=db_masks,
        )

        # 2. Fuse
        retrieved_masks = db_masks[indices] if db_masks is not None else None
        fused_emb = self.fusion(
            query_emb, retrieved, scores,
            retrieved_masks=retrieved_masks,
        )

        # 3. Inject fused embedding into every recycling copy
        num_recycles = batch["seq_embedding"].shape[-1]
        batch["seq_embedding"] = fused_emb.unsqueeze(-1).expand(
            *fused_emb.shape, num_recycles
        )

        # 4. Run frozen OpenFold (grads flow through activations, not params)
        outputs = self.openfold(batch)

        # Stash retrieval metadata for logging / analysis
        outputs["retrieval_scores"] = scores.detach()
        outputs["retrieval_indices"] = indices.detach()

        return outputs
