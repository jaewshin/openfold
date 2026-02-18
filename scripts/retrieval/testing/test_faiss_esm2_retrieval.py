#!/usr/bin/env python3
"""Sanity-check retrieval correctness for an ESM2 FAISS index.

This script samples vectors from merged ESM2 embedding shards (shard_*.npy),
queries the FAISS index, and reports self-retrieval recall.

Default behavior is memory-friendly: it does row-index checks only.
Optional ID-level checks are supported via --labels_pkl.
"""

from __future__ import annotations

import argparse
import bisect
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np


def _discover_shards(embeddings_dir: Path) -> List[Path]:
    shards = sorted(embeddings_dir.glob("shard_*.npy"))
    if not shards:
        raise RuntimeError(f"No shard_*.npy files found under {embeddings_dir}")
    return shards


def _build_layout(shard_paths: Sequence[Path]) -> Tuple[List[int], List[int]]:
    counts: List[int] = []
    cumulative: List[int] = []
    total = 0
    for path in shard_paths:
        n = int(np.load(path, mmap_mode="r").shape[0])
        counts.append(n)
        total += n
        cumulative.append(total)
    return counts, cumulative


def _fetch_vectors_by_global_idx(
    shard_paths: Sequence[Path],
    cumulative: Sequence[int],
    global_indices: np.ndarray,
) -> np.ndarray:
    grouped: Dict[int, List[Tuple[int, int]]] = {}
    for out_pos, gidx in enumerate(global_indices.tolist()):
        shard_i = bisect.bisect_right(cumulative, gidx)
        shard_start = 0 if shard_i == 0 else cumulative[shard_i - 1]
        local_idx = gidx - shard_start
        grouped.setdefault(shard_i, []).append((out_pos, local_idx))

    dim = int(np.load(shard_paths[0], mmap_mode="r").shape[1])
    out = np.empty((len(global_indices), dim), dtype=np.float32)

    for shard_i, pairs in grouped.items():
        mmap = np.load(shard_paths[shard_i], mmap_mode="r")
        local_rows = np.array([p[1] for p in pairs], dtype=np.int64)
        chunk = np.array(mmap[local_rows], dtype=np.float32, copy=True)
        for (out_pos, _), vec in zip(pairs, chunk):
            out[out_pos] = vec
    return out


def _load_labels(labels_pkl: Path, expected: int) -> List[str]:
    with open(labels_pkl, "rb") as f:
        labels = pickle.load(f)
    if len(labels) != expected:
        raise RuntimeError(
            f"labels count mismatch: {len(labels)} != expected {expected} "
            f"(from index.ntotal)"
        )
    return labels


def _row_hits(neighbors: np.ndarray, truth_rows: np.ndarray) -> Tuple[int, int]:
    valid = neighbors >= 0
    row_eq = neighbors == truth_rows[:, None]
    row_eq = row_eq & valid
    hit1 = int(row_eq[:, 0].sum()) if row_eq.shape[1] > 0 else 0
    hitk = int(row_eq.any(axis=1).sum())
    return hit1, hitk


def _id_hits(neighbors: np.ndarray, truth_rows: np.ndarray, labels: Sequence[str]) -> Tuple[int, int]:
    hit1 = 0
    hitk = 0
    for i in range(neighbors.shape[0]):
        true_id = labels[int(truth_rows[i])]
        topk = [int(x) for x in neighbors[i].tolist() if int(x) >= 0]
        if topk:
            top1_id = labels[topk[0]]
            if top1_id == true_id:
                hit1 += 1
        if any(labels[r] == true_id for r in topk):
            hitk += 1
    return hit1, hitk


def _default_labels_path(index_file: Path) -> Path:
    stem = str(index_file)
    if stem.endswith(".index"):
        stem = stem[: -len(".index")]
    elif stem.endswith(".faiss"):
        stem = stem[: -len(".faiss")]
    return Path(stem + "_labels.pkl")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate ESM2 FAISS index retrieval")
    parser.add_argument("--index_file", required=True, help="Path to FAISS .index file")
    parser.add_argument(
        "--embeddings_dir",
        required=True,
        help="Directory containing merged shard_*.npy files",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=2000,
        help="Number of query vectors to sample",
    )
    parser.add_argument(
        "--sample_shard",
        type=int,
        default=-1,
        help="If >=0, sample only from this shard index for a fast test",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k to retrieve")
    parser.add_argument("--nprobe", type=int, default=0, help="Override index nprobe if IVF")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--normalize_queries",
        action="store_true",
        help="L2-normalize sampled queries before search",
    )
    parser.add_argument(
        "--labels_pkl",
        default="",
        help="Optional labels .pkl for ID-level metrics (default: infer from index_file)",
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=5,
        help="Number of query/top1 examples to print",
    )
    parser.add_argument(
        "--skip_ntotal_check",
        action="store_true",
        help="Skip strict check that total vectors in shards == index.ntotal",
    )
    parser.add_argument(
        "--min_recall_at_1",
        type=float,
        default=-1.0,
        help="Optional failure threshold for row recall@1",
    )
    parser.add_argument(
        "--min_recall_at_k",
        type=float,
        default=-1.0,
        help="Optional failure threshold for row recall@k",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()

    index_file = Path(args.index_file)
    embeddings_dir = Path(args.embeddings_dir)

    print(f"[info] Loading index: {index_file}")
    index = faiss.read_index(str(index_file))
    if args.nprobe > 0 and hasattr(index, "nprobe"):
        index.nprobe = args.nprobe
    print(f"[info] index.ntotal={int(index.ntotal)} dim={int(index.d)}")
    if hasattr(index, "nprobe"):
        print(f"[info] nprobe={int(index.nprobe)}")

    shard_paths = _discover_shards(embeddings_dir)
    counts, cumulative = _build_layout(shard_paths)
    shard_total = cumulative[-1]
    print(f"[info] found {len(shard_paths)} shards, total_vectors={shard_total}")

    if not args.skip_ntotal_check and shard_total != int(index.ntotal):
        raise RuntimeError(
            f"Vector count mismatch: shards={shard_total} vs index.ntotal={int(index.ntotal)}"
        )

    searchable_total = min(shard_total, int(index.ntotal))
    if searchable_total <= 0:
        raise RuntimeError("No vectors available for validation")

    rng = np.random.default_rng(args.seed)
    if args.sample_shard >= 0:
        if args.sample_shard >= len(shard_paths):
            raise RuntimeError(
                f"--sample_shard={args.sample_shard} out of range [0, {len(shard_paths)-1}]"
            )
        shard_n = counts[args.sample_shard]
        n = min(args.sample_size, shard_n)
        local = rng.choice(shard_n, size=n, replace=False)
        base = 0 if args.sample_shard == 0 else cumulative[args.sample_shard - 1]
        sample_rows = base + local
        print(f"[info] sampling {n} queries from shard index {args.sample_shard}")
    else:
        n = min(args.sample_size, searchable_total)
        sample_rows = rng.choice(searchable_total, size=n, replace=False)
        print(f"[info] sampling {n} queries across all shards")

    queries = _fetch_vectors_by_global_idx(shard_paths, cumulative, sample_rows)
    if queries.shape[1] != int(index.d):
        raise RuntimeError(
            f"Dim mismatch: queries={queries.shape[1]} vs index={int(index.d)}"
        )

    if args.normalize_queries:
        faiss.normalize_L2(queries)
        print("[info] normalized queries with L2")

    print(f"[info] running index.search(k={args.k})")
    distances, neighbors = index.search(queries, int(args.k))

    row_hit1, row_hitk = _row_hits(neighbors, sample_rows)
    row_r1 = row_hit1 / n
    row_rk = row_hitk / n
    print(f"[result] row_recall@1={row_r1:.4f} ({row_hit1}/{n})")
    print(f"[result] row_recall@{args.k}={row_rk:.4f} ({row_hitk}/{n})")

    labels: Optional[List[str]] = None
    labels_path = Path(args.labels_pkl) if args.labels_pkl else _default_labels_path(index_file)
    if labels_path.exists():
        print(f"[info] loading labels: {labels_path}")
        labels = _load_labels(labels_path, int(index.ntotal))
        id_hit1, id_hitk = _id_hits(neighbors, sample_rows, labels)
        print(f"[result] id_recall@1={id_hit1 / n:.4f} ({id_hit1}/{n})")
        print(f"[result] id_recall@{args.k}={id_hitk / n:.4f} ({id_hitk}/{n})")
    else:
        print(f"[warn] labels file not found, skipping ID-level metrics: {labels_path}")

    show_n = min(int(args.show_examples), n)
    if show_n > 0:
        print("[examples]")
        for i in range(show_n):
            q_row = int(sample_rows[i])
            top_row = int(neighbors[i, 0]) if neighbors.shape[1] > 0 else -1
            top_score = float(distances[i, 0]) if top_row >= 0 else float("nan")
            if labels is not None and top_row >= 0:
                q_name = labels[q_row]
                t_name = labels[top_row]
            else:
                q_name = f"row:{q_row}"
                t_name = f"row:{top_row}"
            status = "OK" if q_name == t_name else "MISS"
            print(
                f"  {i+1}. query={q_name} top1={t_name} "
                f"score={top_score:.6f} [{status}]"
            )

    if args.min_recall_at_1 >= 0 and row_r1 < args.min_recall_at_1:
        raise SystemExit(
            f"row_recall@1={row_r1:.4f} below threshold {args.min_recall_at_1:.4f}"
        )
    if args.min_recall_at_k >= 0 and row_rk < args.min_recall_at_k:
        raise SystemExit(
            f"row_recall@{args.k}={row_rk:.4f} below threshold {args.min_recall_at_k:.4f}"
        )


if __name__ == "__main__":
    main()
