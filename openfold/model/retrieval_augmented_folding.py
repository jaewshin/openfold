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
    """Wraps a frozen OpenFold-SoloSeq with trainable retrieval + fusion heads.

    Forward flow:
        1. Sequence retriever/fusion path (optional).
        2. Structure retriever/fusion path (optional).
        3. Weighted combine of available fused embeddings.
        4. The fused embedding replaces ``feats["seq_embedding"]`` in the
           batch and is passed through the frozen ``AlphaFold`` model.

    Because the AlphaFold parameters have ``requires_grad=False`` but the
    forward pass still builds a computation graph, gradients from the
    structural loss flow back through the frozen activations into the
    retriever and fusion parameters.

    Args:
        config:          Full OpenFold config (should be a seqemb preset).
        emb_dim:         Dimension of pre-computed embeddings (1280 for ESM-1b).
        retriever_proj:  Projection dimension for both retrievers.
        top_k:           Number of database entries to retrieve per source.
        fusion_heads:    Number of attention heads in each fusion module.
        fusion_dropout:  Dropout in each fusion module.
        seq_weight:      Initial mixture weight for sequence database path.
        struct_weight:   Initial mixture weight for structure database path.
    """

    def __init__(
        self,
        config,
        emb_dim: int = 1280,
        retriever_proj: int = 128,
        top_k: int = 16,
        fusion_heads: int = 8,
        fusion_dropout: float = 0.0,
        seq_weight: float = 0.5,
        struct_weight: float = 0.5,
        retrieval_ablation: str = "both",
    ):
        super().__init__()

        # --- Frozen backbone --------------------------------------------------
        self.openfold = AlphaFold(config)
        self.openfold.eval()
        for p in self.openfold.parameters():
            p.requires_grad_(False)

        # --- Trainable modules ------------------------------------------------
        self.seq_retriever = EmbeddingRetriever(
            emb_dim=emb_dim,
            c_proj=retriever_proj,
            top_k=top_k,
        )
        self.struct_retriever = EmbeddingRetriever(
            emb_dim=emb_dim,
            c_proj=retriever_proj,
            top_k=top_k,
        )
        self.seq_fusion = CrossAttentionFusion(
            emb_dim=emb_dim,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
        )
        self.struct_fusion = CrossAttentionFusion(
            emb_dim=emb_dim,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
        )
        init_w = torch.tensor([max(seq_weight, 1e-8), max(struct_weight, 1e-8)], dtype=torch.float32)
        self.db_mix_logits = nn.Parameter(init_w.log())
        self.retrieval_ablation = retrieval_ablation
        if self.retrieval_ablation not in {"both", "seq_only", "struct_only"}:
            raise ValueError(
                f"Unsupported retrieval_ablation={retrieval_ablation!r}. "
                "Expected one of {'both', 'seq_only', 'struct_only'}."
            )

        # Backward-compat aliases for existing code/tests expecting single path names.
        self.retriever = self.seq_retriever
        self.fusion = self.seq_fusion

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Yield only the parameters that should be optimised."""
        yield from self.seq_retriever.parameters()
        yield from self.struct_retriever.parameters()
        yield from self.seq_fusion.parameters()
        yield from self.struct_fusion.parameters()
        yield self.db_mix_logits

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: dict,
        db_embs: Optional[torch.Tensor] = None,
        db_masks: Optional[torch.Tensor] = None,
        struct_db_embs: Optional[torch.Tensor] = None,
        struct_db_masks: Optional[torch.Tensor] = None,
    ) -> dict:
        """Run retrieval-augmented folding.

        Args:
            batch:    Standard OpenFold feature dict (with recycling dim).
                      Must contain ``"seq_embedding"`` (SoloSeq mode).
            db_embs:          [M, N_d, D] optional sequence embedding DB.
            db_masks:         [M, N_d]    optional sequence DB residue mask.
            struct_db_embs:   [M2, N_s, D] optional structure embedding DB.
            struct_db_masks:  [M2, N_s]    optional structure DB residue mask.

        Returns:
            outputs:  OpenFold output dict (same as ``AlphaFold.forward``).
                      Also includes source-specific retrieval metadata and
                      combine weights for analysis.
        """
        # The query embedding is stored with a recycling dim at the end.
        # We use the first recycling copy (they are all the same for
        # seq_embedding since it doesn't change across recycles).
        query_emb = batch["seq_embedding"][..., 0]  # [N_q, D]
        query_mask = batch["seq_mask"][..., 0] if "seq_mask" in batch else None

        use_seq = self.retrieval_ablation in {"both", "seq_only"}
        use_struct = self.retrieval_ablation in {"both", "struct_only"}

        if (not use_seq or db_embs is None) and (not use_struct or struct_db_embs is None):
            raise ValueError("At least one database must be provided: db_embs and/or struct_db_embs")

        seq_fused = None
        seq_scores = None
        seq_indices = None
        if use_seq and db_embs is not None:
            seq_scores, seq_indices, seq_retrieved = self.seq_retriever(
                query_emb,
                db_embs,
                query_mask=query_mask,
                db_masks=db_masks,
            )
            seq_retrieved_masks = db_masks[seq_indices] if db_masks is not None else None
            seq_fused = self.seq_fusion(
                query_emb,
                seq_retrieved,
                seq_scores,
                retrieved_masks=seq_retrieved_masks,
            )

        struct_fused = None
        struct_scores = None
        struct_indices = None
        if use_struct and struct_db_embs is not None:
            struct_scores, struct_indices, struct_retrieved = self.struct_retriever(
                query_emb,
                struct_db_embs,
                query_mask=query_mask,
                db_masks=struct_db_masks,
            )
            struct_retrieved_masks = (
                struct_db_masks[struct_indices] if struct_db_masks is not None else None
            )
            struct_fused = self.struct_fusion(
                query_emb,
                struct_retrieved,
                struct_scores,
                retrieved_masks=struct_retrieved_masks,
            )

        if seq_fused is not None and struct_fused is not None:
            mix = torch.softmax(self.db_mix_logits, dim=0)
            fused_emb = mix[0] * seq_fused + mix[1] * struct_fused
            outputs_mix = mix.detach()
        elif seq_fused is not None:
            fused_emb = seq_fused
            outputs_mix = torch.tensor([1.0, 0.0], device=query_emb.device)
        else:
            fused_emb = struct_fused
            outputs_mix = torch.tensor([0.0, 1.0], device=query_emb.device)

        # 3. Inject fused embedding into every recycling copy
        num_recycles = batch["seq_embedding"].shape[-1]
        batch["seq_embedding"] = fused_emb.unsqueeze(-1).expand(
            *fused_emb.shape, num_recycles
        )

        # 4. Run frozen OpenFold (grads flow through activations, not params)
        outputs = self.openfold(batch)

        # Stash retrieval metadata for logging / analysis
        if seq_scores is not None:
            outputs["retrieval_scores"] = seq_scores.detach()  # backward-compatible key
            outputs["retrieval_indices"] = seq_indices.detach()  # backward-compatible key
            outputs["seq_retrieval_scores"] = seq_scores.detach()
            outputs["seq_retrieval_indices"] = seq_indices.detach()
        if struct_scores is not None:
            outputs["struct_retrieval_scores"] = struct_scores.detach()
            outputs["struct_retrieval_indices"] = struct_indices.detach()
        outputs["retrieval_source_weights"] = outputs_mix

        return outputs
