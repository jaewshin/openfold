#!/usr/bin/env python3
"""Merge per-job embedding chunk outputs into one embedding directory.

Each chunk directory is expected to contain run_faiss.py embedding shards:
  shard_00000.npy
  shard_00000.pkl
  ...

The merged output is renumbered globally:
  shard_00000.npy/.pkl
  shard_00001.npy/.pkl
  ...
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from glob import glob
from pathlib import Path
from typing import List, Tuple


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        rel = os.path.relpath(src, dst.parent)
        os.symlink(rel, dst)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            # Cross-filesystem fallback.
            shutil.copy2(src, dst)
        return
    raise ValueError(f"Unsupported mode={mode}")


def _labels_count(pkl_path: Path) -> int:
    with open(pkl_path, "rb") as f:
        labels = pickle.load(f)
    return len(labels)


def _discover_chunk_dirs(chunks_root: Path) -> List[Path]:
    # Most generated directories are chunk_XXXXX; keep deterministic order.
    dirs = sorted([p for p in chunks_root.glob("chunk_*") if p.is_dir()])
    if dirs:
        return dirs
    # Fallback: any directory that contains shard_*.npy
    out = []
    for p in sorted([x for x in chunks_root.iterdir() if x.is_dir()]):
        if list(p.glob("shard_*.npy")):
            out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser(description="Merge embedding shards from parallel jobs")
    parser.add_argument("--chunks_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="How to materialize merged shards",
    )
    parser.add_argument(
        "--allow_missing_pairs",
        action="store_true",
        help="Skip shards missing matching .pkl/.npy instead of failing",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dirs = _discover_chunk_dirs(args.chunks_root)
    if not chunk_dirs:
        raise SystemExit(f"No chunk dirs found under {args.chunks_root}")

    merged_count = 0
    total_labels = 0
    source_map: List[Tuple[str, str, str]] = []

    for chunk_dir in chunk_dirs:
        npy_files = sorted(glob(str(chunk_dir / "shard_*.npy")))
        for npy in npy_files:
            npy_path = Path(npy)
            pkl_path = npy_path.with_suffix(".pkl")
            if not pkl_path.exists():
                msg = f"Missing label file for {npy_path}"
                if args.allow_missing_pairs:
                    print(f"[warn] {msg}; skipping")
                    continue
                raise SystemExit(msg)

            dst_npy = args.output_dir / f"shard_{merged_count:05d}.npy"
            dst_pkl = args.output_dir / f"shard_{merged_count:05d}.pkl"
            _link_or_copy(npy_path, dst_npy, args.mode)
            _link_or_copy(pkl_path, dst_pkl, args.mode)

            n = _labels_count(pkl_path)
            total_labels += n
            source_map.append((str(dst_npy.name), str(npy_path), str(pkl_path)))
            merged_count += 1

    map_path = args.output_dir / "source_map.tsv"
    with open(map_path, "w") as f:
        f.write("merged_shard\tsource_npy\tsource_pkl\n")
        for row in source_map:
            f.write("\t".join(row) + "\n")

    meta = {
        "chunks_root": str(args.chunks_root),
        "output_dir": str(args.output_dir),
        "mode": args.mode,
        "num_chunk_dirs": len(chunk_dirs),
        "num_merged_shards": merged_count,
        "total_labels": total_labels,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

