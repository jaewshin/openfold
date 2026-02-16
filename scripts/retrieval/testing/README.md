# Retrieval Pipeline Testing Scripts

This directory contains runnable scripts implementing the retrieval-augmented
pipeline test plan.

## 1) Validate dataset fixture

```bash
python scripts/retrieval/testing/validate_retrieval_fixture.py --dataset_dir /path/to/rag_data
```

Expected dataset layout:

- `metadata.json`
- `splits.json`
- `train.fasta`
- `val.fasta`
- `structures/*.cif` or `structures/*.cif.gz`

## 2) Module-level tests

```bash
python scripts/retrieval/testing/test_retrieval_fusion_modules.py
```

Validates:

- `EmbeddingRetriever` output shapes and score normalization
- `top_k > M` behavior
- `CrossAttentionFusion` finite output and gate sensitivity

## 3) Wrapper smoke test with manual retrieval

```bash
python scripts/retrieval/testing/smoke_test_retrieval_augmented_wrapper.py
```

This patches in a lightweight fake AlphaFold backbone and checks:

- retrieval metadata is returned (`retrieval_scores`, `retrieval_indices`)
- fused embedding is injected across recycles
- gradients flow to retriever/fusion
- frozen OpenFold backbone remains gradient-free

## Suggested order

1. Run fixture validation.
2. Run module-level tests.
3. Run wrapper smoke test.
