#!/usr/bin/env python3
"""Deterministic tests for EmbeddingRetriever and CrossAttentionFusion."""

import torch

from openfold.model.retrieval_fusion import CrossAttentionFusion, EmbeddingRetriever


def assert_close(x: torch.Tensor, y: torch.Tensor, atol=1e-5, rtol=1e-5, msg=""):
    if not torch.allclose(x, y, atol=atol, rtol=rtol):
        raise AssertionError(msg or f"Tensors not close. max_diff={(x - y).abs().max().item()}")


def test_embedding_retriever():
    torch.manual_seed(7)
    N_q, M, N_d, D = 11, 5, 13, 32

    retriever = EmbeddingRetriever(emb_dim=D, c_proj=16, top_k=4)

    query = torch.randn(N_q, D)
    db = torch.randn(M, N_d, D)

    # Make db[2] very similar to query so it should rank highly.
    db[2, :N_q] = query + 0.01 * torch.randn_like(query)

    query_mask = torch.ones(N_q)
    db_masks = torch.ones(M, N_d)
    db_masks[1, -4:] = 0  # partial mask stress

    scores, indices, retrieved = retriever(query, db, query_mask=query_mask, db_masks=db_masks)

    assert scores.shape == (4,), f"scores shape mismatch: {scores.shape}"
    assert indices.shape == (4,), f"indices shape mismatch: {indices.shape}"
    assert retrieved.shape == (4, N_d, D), f"retrieved shape mismatch: {retrieved.shape}"
    assert_close(scores.sum(), torch.tensor(1.0), atol=1e-6, msg="scores should sum to 1")

    # top_k > M behavior
    retriever2 = EmbeddingRetriever(emb_dim=D, c_proj=8, top_k=99)
    scores2, indices2, retrieved2 = retriever2(query, db)
    assert scores2.shape[0] == M, "Expected K=min(top_k, M)"
    assert indices2.shape[0] == M
    assert retrieved2.shape[0] == M


def test_cross_attention_fusion():
    torch.manual_seed(13)
    N_q, K, N_kv, D, H = 9, 3, 7, 32, 4

    fusion = CrossAttentionFusion(emb_dim=D, num_heads=H, dropout=0.0)
    query = torch.randn(N_q, D)
    retrieved = torch.randn(K, N_kv, D)
    scores = torch.softmax(torch.randn(K), dim=0)
    masks = torch.ones(K, N_kv)

    out_default_gate = fusion(query, retrieved, scores, retrieved_masks=masks)
    assert out_default_gate.shape == query.shape
    assert torch.isfinite(out_default_gate).all(), "NaN/Inf found in fused output"

    # Increase gate and verify larger deviation from query than default gate case.
    with torch.no_grad():
        fusion.gate.fill_(8.0)
    out_high_gate = fusion(query, retrieved, scores, retrieved_masks=masks)

    delta_default = (out_default_gate - query).norm().item()
    delta_high = (out_high_gate - query).norm().item()
    if not delta_high > delta_default:
        raise AssertionError(
            f"Expected stronger fusion effect with high gate: delta_high={delta_high}, delta_default={delta_default}"
        )

    # Mask sanity: all-zero mask should either error or produce finite output.
    all_zero_masks = torch.zeros(K, N_kv)
    try:
        out = fusion(query, retrieved, scores, retrieved_masks=all_zero_masks)
        if not torch.isfinite(out).all():
            raise AssertionError("all-zero mask produced non-finite output")
    except RuntimeError:
        # Acceptable: explicit failure mode from attention softmax over -inf.
        pass


def main():
    test_embedding_retriever()
    test_cross_attention_fusion()
    print("[OK] retrieval_fusion module tests passed")


if __name__ == "__main__":
    main()
