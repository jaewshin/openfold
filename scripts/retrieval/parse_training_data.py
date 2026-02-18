#!/usr/bin/env python3
"""Parse retrieval training data into a normalized JSONL manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Allow running directly from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.retrieval.retrieval_data import build_manifest, write_manifest_jsonl


def main():
    parser = argparse.ArgumentParser(
        description="Parse download_structures.py output into a training manifest",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Path containing splits.json, train.fasta, val.fasta, structures/",
    )
    parser.add_argument(
        "--output_manifest",
        type=Path,
        default=None,
        help="Output JSONL path (default: <dataset_dir>/manifest.jsonl)",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Skip rows with missing sequence/structure instead of failing",
    )
    args = parser.parse_args()

    records = build_manifest(args.dataset_dir, strict=(not args.allow_missing))
    output_path = args.output_manifest or (args.dataset_dir / "manifest.jsonl")
    write_manifest_jsonl(records, output_path)

    split_counts = Counter(r.split for r in records)
    summary = {
        "manifest_path": str(output_path),
        "total": len(records),
        "train": int(split_counts.get("train", 0)),
        "val": int(split_counts.get("val", 0)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
