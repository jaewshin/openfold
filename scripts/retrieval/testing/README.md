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
python scripts/retrieval/testing/test_retrieval_data_parser.py
python scripts/retrieval/testing/test_packed_retrieval_dataset.py
python scripts/retrieval/testing/test_retrieval_fusion_modules.py
python scripts/retrieval/testing/smoke_test_retrieval_augmented_wrapper.py
python scripts/retrieval/testing/test_train_retrieval_lightning.py
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

## 3) Data parser tests

Command:

```bash
python scripts/retrieval/testing/test_retrieval_data_parser.py
```

Validates:

- `download_structures.py`-style fixture parsing into a normalized manifest
- JSONL manifest read/write roundtrip

## 4) Wrapper smoke test with manual retrieval

Command:

```bash
python scripts/retrieval/testing/smoke_test_retrieval_augmented_wrapper.py
```

This patches in a lightweight fake AlphaFold backbone and checks:

- retrieval metadata is returned (`retrieval_scores`, `retrieval_indices`)
- fused embedding is injected across recycles
- gradients flow to retriever/fusion
- frozen OpenFold backbone remains gradient-free

## 5) Packed dataset format tests

Command:

```bash
python scripts/retrieval/testing/test_packed_retrieval_dataset.py
```

Validates:

- sharded packed feature writing
- packed split loading with lazy shard cache
- packed dataset integration with `RetrievalDataModule`

## 6) Lightning training smoke test

Command:

```bash
python scripts/retrieval/testing/test_train_retrieval_lightning.py
```

Validates:

- retrieval-augmented Lightning `training_step` computes a loss
- optimizer step updates trainable retrieval/fusion parameters
- fake backbone + fake loss wiring for low-memory CI smoke checks

## Suggested order

1. Run fixture validation.
2. Run parser tests.
3. Run packed dataset tests.
4. Run module-level tests.
5. Run wrapper smoke test.
6. Run Lightning training smoke test.
