#!/usr/bin/env python3
"""Smoke test RetrievalAugmentedFolding with manual retrieval candidates.

This script patches the internal AlphaFold backbone with a lightweight fake
module so you can validate retrieval/fusion wiring and gradient ownership
without requiring full OpenFold runtime setup.
"""

import sys
from pathlib import Path
from unittest.mock import patch

# Allow running this file directly from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from openfold.model.retrieval_augmented_folding import RetrievalAugmentedFolding


class FakeAlphaFold(nn.Module):
    def __init__(self, _config):
        super().__init__()
        self.backbone_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        seq_emb = batch["seq_embedding"]  # [N, D, R]
        # Depend on seq_embedding so gradients can flow to retriever/fusion.
        loss_tensor = (seq_emb * self.backbone_scale).pow(2).mean()
        return {
            "smoke_loss": loss_tensor,
            "final_atom_positions": torch.zeros(seq_emb.shape[0], 37, 3, device=seq_emb.device),
        }


def _build_batch(N=17, D=32, R=3):
    return {
        "seq_embedding": torch.randn(N, D, R, requires_grad=True),
        "seq_mask": torch.ones(N, R),
    }


def main():
    torch.manual_seed(1234)

    with patch("openfold.model.retrieval_augmented_folding.AlphaFold", FakeAlphaFold):
        model = RetrievalAugmentedFolding(
            config={},
            emb_dim=32,
            retriever_proj=8,
            top_k=2,
            fusion_heads=4,
            fusion_dropout=0.0,
        )
        seq_only_model = RetrievalAugmentedFolding(
            config={},
            emb_dim=32,
            retriever_proj=8,
            top_k=2,
            fusion_heads=4,
            fusion_dropout=0.0,
            retrieval_ablation="seq_only",
        )
        struct_only_model = RetrievalAugmentedFolding(
            config={},
            emb_dim=32,
            retriever_proj=8,
            top_k=2,
            fusion_heads=4,
            fusion_dropout=0.0,
            retrieval_ablation="struct_only",
        )

    # OpenFold params must be frozen by wrapper ctor
    for p in model.openfold.parameters():
        if p.requires_grad:
            raise AssertionError("OpenFold parameter unexpectedly requires grad")

    batch = _build_batch(N=17, D=32, R=3)

    # Manual retrieval DB: include query-identical candidate plus distractors
    query_emb = batch["seq_embedding"][..., 0].detach()
    db_embs = torch.stack(
        [
            query_emb.clone(),
            torch.randn_like(query_emb),
            torch.randn_like(query_emb),
        ],
        dim=0,
    )  # [M, N, D]
    db_masks = torch.ones(db_embs.shape[0], db_embs.shape[1])

    struct_db_embs = torch.stack(
        [
            query_emb.clone() + 0.01 * torch.randn_like(query_emb),
            torch.randn_like(query_emb),
            torch.randn_like(query_emb),
        ],
        dim=0,
    )
    struct_db_masks = torch.ones(struct_db_embs.shape[0], struct_db_embs.shape[1])

    outputs = model(
        batch=batch,
        db_embs=db_embs,
        db_masks=db_masks,
        struct_db_embs=struct_db_embs,
        struct_db_masks=struct_db_masks,
    )
    _ = seq_only_model(batch=_build_batch(N=17, D=32, R=3), db_embs=db_embs, db_masks=db_masks)
    _ = struct_only_model(
        batch=_build_batch(N=17, D=32, R=3),
        struct_db_embs=struct_db_embs,
        struct_db_masks=struct_db_masks,
    )

    # Output contract checks
    assert "retrieval_scores" in outputs
    assert "retrieval_indices" in outputs
    assert "seq_retrieval_scores" in outputs
    assert "seq_retrieval_indices" in outputs
    assert "struct_retrieval_scores" in outputs
    assert "struct_retrieval_indices" in outputs
    assert "retrieval_source_weights" in outputs
    assert "smoke_loss" in outputs
    assert "final_atom_positions" in outputs

    # Recycle replacement check: every recycle copy should be identical after injection.
    seq_embedding = batch["seq_embedding"]
    for r in range(1, seq_embedding.shape[-1]):
        if not torch.allclose(seq_embedding[..., 0], seq_embedding[..., r]):
            raise AssertionError("seq_embedding recycle copies are not identical after fusion injection")

    # Gradient flow check
    loss = outputs["smoke_loss"]
    loss.backward()

    retriever_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.retriever.parameters())
    fusion_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.fusion.parameters())
    struct_retriever_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.struct_retriever.parameters()
    )
    struct_fusion_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.struct_fusion.parameters()
    )
    if not retriever_grad:
        raise AssertionError("No retriever gradients found")
    if not fusion_grad:
        raise AssertionError("No fusion gradients found")
    if not struct_retriever_grad:
        raise AssertionError("No structure retriever gradients found")
    if not struct_fusion_grad:
        raise AssertionError("No structure fusion gradients found")

    # Frozen backbone should not accumulate grads
    for p in model.openfold.parameters():
        if p.grad is not None:
            raise AssertionError("Frozen OpenFold backbone unexpectedly received gradients")

    print("[OK] RetrievalAugmentedFolding wrapper smoke test passed")


if __name__ == "__main__":
    main()
