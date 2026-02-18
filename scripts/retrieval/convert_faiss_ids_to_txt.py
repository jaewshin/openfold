#!/usr/bin/env python3
"""Convert FAISS row-id mappings to line-based text (one id per row).

Supported inputs:
- `.pkl` list of labels (from run_faiss.py)
- `.txt` / `.txt.gz` line-based ids

Output:
- `.txt` where line i maps to FAISS row i.
"""

from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path


def _normalize_id(value: str) -> str:
    value = str(value).strip()
    return value.split()[0] if value else value


def convert(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffixes = input_path.suffixes
    if suffixes and suffixes[-1] == ".pkl":
        with open(input_path, "rb") as f:
            labels = pickle.load(f)
        with open(output_path, "w") as out:
            for item in labels:
                out.write(_normalize_id(str(item)) + "\n")
        return

    if suffixes[-1:] == [".gz"]:
        in_fh = gzip.open(input_path, "rt")
    else:
        in_fh = open(input_path, "r")

    with in_fh as src, open(output_path, "w") as out:
        for line in src:
            if not line.strip():
                out.write("\n")
                continue
            out.write(_normalize_id(line) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FAISS id map to .txt (line-aligned with FAISS rows)")
    parser.add_argument("--input_path", type=Path, required=True, help="Input mapping (.pkl/.txt/.txt.gz)")
    parser.add_argument("--output_path", type=Path, required=True, help="Output .txt path")
    args = parser.parse_args()

    convert(args.input_path, args.output_path)
    print(f"[OK] Wrote id map: {args.output_path}")


if __name__ == "__main__":
    main()
