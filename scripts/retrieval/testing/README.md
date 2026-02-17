# Retrieval Pipeline Testing Scripts

This directory contains runnable scripts implementing the retrieval-augmented
pipeline test plan.

## Prerequisites

Run from the repository root:

```bash
cd /workspace/openfold
```

Make sure your Python environment has dependencies installed (at minimum `torch` for module/wrapper tests):

```bash
python -c "import torch; print(torch.__version__)"
```

## Quick run (all scripts)

You can also run from inside `scripts/retrieval/testing/` now; the scripts auto-add the repo root to `PYTHONPATH`.

```bash
cd /workspace/openfold/scripts/retrieval/testing
python test_retrieval_fusion_modules.py
python smoke_test_retrieval_augmented_wrapper.py
```


Set your downloaded dataset path once, then run all checks:

```bash
DATASET_DIR=/absolute/path/to/rag_data

python scripts/retrieval/testing/validate_retrieval_fixture.py --dataset_dir "$DATASET_DIR"
python scripts/retrieval/testing/test_retrieval_fusion_modules.py
python scripts/retrieval/testing/smoke_test_retrieval_augmented_wrapper.py
```

## 1) Validate dataset fixture

Command:

```bash
python scripts/retrieval/testing/validate_retrieval_fixture.py --dataset_dir /absolute/path/to/rag_data
```

Optional fail-fast mode:

```bash
python scripts/retrieval/testing/validate_retrieval_fixture.py --dataset_dir /absolute/path/to/rag_data --fail_fast
```

Expected dataset layout under `--dataset_dir`:

- `metadata.json`
- `splits.json`
- `train.fasta`
- `val.fasta`
- `structures/*.cif` or `structures/*.cif.gz`

## 2) Module-level tests

Command:

```bash
python scripts/retrieval/testing/test_retrieval_fusion_modules.py
```

Validates:

- `EmbeddingRetriever` output shapes and score normalization
- `top_k > M` behavior
- `CrossAttentionFusion` finite output and gate sensitivity

## 3) Wrapper smoke test with manual retrieval

Command:

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
