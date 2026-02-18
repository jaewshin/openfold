#!/usr/bin/env python3
"""Build a packed, preprocessed retrieval dataset for faster training.

This avoids repeated FASTA/mmCIF parsing at train time by materializing
OpenFold-ready feature tensors into sharded `.pt` files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.retrieval.retrieval_data import build_packed_feature_dataset


def main():
    parser = argparse.ArgumentParser(description="Build packed retrieval feature dataset")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, default=None)
    parser.add_argument("--manifest_path", type=Path, default=None)
    parser.add_argument("--seq_embedding_dir", type=Path, default=None)
    parser.add_argument("--config_preset", type=str, default="seqemb_initial_training")
    parser.add_argument("--seq_embedding_dim", type=int, default=1280)
    parser.add_argument("--shard_size", type=int, default=64)
    parser.add_argument("--max_recycling_iters", type=int, default=0)
    parser.add_argument("--strict_seq_embeddings", action="store_true")
    parser.add_argument("--float16_storage", action="store_true")
    args = parser.parse_args()

    if args.dataset_dir is None and args.manifest_path is None:
        raise SystemExit("Provide either --dataset_dir or --manifest_path")

    meta = build_packed_feature_dataset(
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        manifest_path=args.manifest_path,
        seq_embedding_dir=args.seq_embedding_dir,
        config_preset=args.config_preset,
        max_recycling_iters=args.max_recycling_iters,
        strict_seq_embeddings=args.strict_seq_embeddings,
        seq_embedding_dim=args.seq_embedding_dim,
        shard_size=args.shard_size,
        float16_storage=args.float16_storage,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

