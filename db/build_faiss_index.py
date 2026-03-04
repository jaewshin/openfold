from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def build_ids_offsets(ids_path: str | Path, output_offsets_path: str | Path) -> None:
    offsets = []
    cur = 0
    with open(ids_path, "rb") as handle:
        for line in handle:
            offsets.append(cur)
            cur += len(line)

    arr = np.asarray(offsets, dtype=np.uint64)
    Path(output_offsets_path).parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(output_offsets_path)


def build_faiss_index(
    *,
    embeddings_path: str | Path,
    output_index_path: str | Path,
    index_type: str,
    normalize: bool,
    add_batch_size: int,
) -> Tuple[int, int]:
    try:
        import faiss
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError("FAISS is required for build_faiss_index.py") from exc

    emb = np.load(embeddings_path, mmap_mode="r")
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be rank-2, got shape {emb.shape}")

    n, dim = emb.shape
    if index_type == "flat_ip":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "flat_l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unsupported index_type={index_type}")

    for start in range(0, n, add_batch_size):
        batch = np.array(emb[start : start + add_batch_size], copy=True)
        if normalize:
            faiss.normalize_L2(batch)
        index.add(batch)

    output_index_path = Path(output_index_path)
    output_index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_index_path))
    return n, dim


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a FAISS index from precomputed embeddings")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output_index", required=True)

    parser.add_argument("--ids", default=None, help="Optional ids text file")
    parser.add_argument("--output_ids_offsets", default=None)

    parser.add_argument("--index_type", default="flat_ip", choices=["flat_ip", "flat_l2"])
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.add_argument("--add_batch_size", type=int, default=200_000)
    return parser


def _run_cli(args: argparse.Namespace) -> None:
    n, dim = build_faiss_index(
        embeddings_path=args.embeddings,
        output_index_path=args.output_index,
        index_type=args.index_type,
        normalize=args.normalize,
        add_batch_size=args.add_batch_size,
    )
    print(f"Wrote index: {args.output_index} (n={n}, dim={dim})")

    if args.ids and args.output_ids_offsets:
        build_ids_offsets(args.ids, args.output_ids_offsets)
        print(f"Wrote ids offsets: {args.output_ids_offsets}")


if __name__ == "__main__":
    _run_cli(_build_arg_parser().parse_args())
