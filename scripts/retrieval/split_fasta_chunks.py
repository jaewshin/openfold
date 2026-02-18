#!/usr/bin/env python3
"""Split a FASTA into many chunk FASTA files for parallel embedding jobs.

Two modes are supported:
1) --num_splits N      : round-robin assignment across N chunks
2) --seqs_per_split K  : sequential chunks with at most K sequences each
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _write_record(fh, header: str, seq: str, width: int = 80) -> None:
    fh.write(f">{header}\n")
    for i in range(0, len(seq), width):
        fh.write(seq[i : i + width] + "\n")


def _iter_fasta(path: Path):
    with open(path, "r") as f:
        header: Optional[str] = None
        chunks: List[str] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


def split_by_num_splits(input_fasta: Path, output_dir: Path, num_splits: int) -> List[Tuple[Path, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"chunk_{i:05d}.fasta" for i in range(num_splits)]
    fhs = [open(p, "w") for p in paths]
    counts = [0 for _ in range(num_splits)]

    try:
        for idx, (header, seq) in enumerate(_iter_fasta(input_fasta)):
            split_idx = idx % num_splits
            _write_record(fhs[split_idx], header, seq)
            counts[split_idx] += 1
    finally:
        for fh in fhs:
            fh.close()

    out: List[Tuple[Path, int]] = []
    for p, c in zip(paths, counts):
        if c > 0:
            out.append((p, c))
        else:
            # Remove empty chunk files so downstream submitters do not enqueue empty jobs.
            p.unlink(missing_ok=True)
    return out


def split_by_seqs_per_split(
    input_fasta: Path, output_dir: Path, seqs_per_split: int
) -> List[Tuple[Path, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out: List[Tuple[Path, int]] = []
    chunk_idx = 0
    chunk_count = 0
    fh = open(output_dir / f"chunk_{chunk_idx:05d}.fasta", "w")

    try:
        for header, seq in _iter_fasta(input_fasta):
            if chunk_count >= seqs_per_split:
                fh.close()
                out.append((output_dir / f"chunk_{chunk_idx:05d}.fasta", chunk_count))
                chunk_idx += 1
                chunk_count = 0
                fh = open(output_dir / f"chunk_{chunk_idx:05d}.fasta", "w")
            _write_record(fh, header, seq)
            chunk_count += 1
    finally:
        fh.close()

    if chunk_count > 0:
        out.append((output_dir / f"chunk_{chunk_idx:05d}.fasta", chunk_count))
    return out


def main():
    parser = argparse.ArgumentParser(description="Split FASTA into chunk FASTA files")
    parser.add_argument("--input_fasta", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_splits", type=int, default=0)
    parser.add_argument("--seqs_per_split", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=None, help="Default: <output_dir>/manifest.json")
    args = parser.parse_args()

    if (args.num_splits > 0) == (args.seqs_per_split > 0):
        raise SystemExit("Provide exactly one of --num_splits or --seqs_per_split")

    if args.num_splits > 0:
        chunks = split_by_num_splits(args.input_fasta, args.output_dir, args.num_splits)
        mode = {"type": "num_splits", "value": args.num_splits}
    else:
        chunks = split_by_seqs_per_split(args.input_fasta, args.output_dir, args.seqs_per_split)
        mode = {"type": "seqs_per_split", "value": args.seqs_per_split}

    manifest_path = args.manifest or (args.output_dir / "manifest.json")
    manifest: Dict[str, object] = {
        "input_fasta": str(args.input_fasta),
        "output_dir": str(args.output_dir),
        "mode": mode,
        "num_chunks": len(chunks),
        "total_sequences": int(sum(c for _, c in chunks)),
        "chunks": [{"path": str(p), "num_sequences": int(c)} for p, c in chunks],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"manifest": str(manifest_path), "num_chunks": len(chunks)}, indent=2))


if __name__ == "__main__":
    main()
