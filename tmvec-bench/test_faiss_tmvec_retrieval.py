#!/usr/bin/env python3
"""
Sanity-check retrieval quality for a TMVec FAISS index.

This script verifies that vectors sampled from the original embedding shards
can retrieve their own IDs from the built index.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
from glob import glob
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import faiss
import numpy as np


def open_text(path: str, mode: str = "rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def read_ids_file(path: str) -> List[str]:
    with open_text(path, "rt") as fh:
        return [line.rstrip("\n") for line in fh]


def iter_fasta(path: str) -> Iterator[Tuple[str, str]]:
    with open_text(path, "rt") as fh:
        header: Optional[str] = None
        seq_chunks: List[str] = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, "".join(seq_chunks)


def build_part_layout(embeddings_dir: str) -> Tuple[List[str], List[int]]:
    part_paths = sorted(glob(str(Path(embeddings_dir) / "part_*.npy")))
    if not part_paths:
        raise RuntimeError(f"No part_*.npy found in {embeddings_dir}")

    cum = []
    total = 0
    for p in part_paths:
        n = int(np.load(p, mmap_mode="r").shape[0])
        total += n
        cum.append(total)
    return part_paths, cum


def build_partial_cum(part_paths: List[str], upto_part: int) -> List[int]:
    cum = []
    total = 0
    for i in range(upto_part + 1):
        n = int(np.load(part_paths[i], mmap_mode="r").shape[0])
        total += n
        cum.append(total)
    return cum


def fetch_vectors_by_global_idx(
    part_paths: List[str],
    cum_counts: List[int],
    global_indices: np.ndarray,
) -> np.ndarray:
    grouped: Dict[int, List[Tuple[int, int]]] = {}
    for out_pos, gidx in enumerate(global_indices.tolist()):
        part_i = bisect.bisect_right(cum_counts, gidx)
        part_start = 0 if part_i == 0 else cum_counts[part_i - 1]
        local_idx = gidx - part_start
        grouped.setdefault(part_i, []).append((out_pos, local_idx))

    d = int(np.load(part_paths[0], mmap_mode="r").shape[1])
    out = np.empty((len(global_indices), d), dtype=np.float32)

    for part_i, pairs in grouped.items():
        mmap = np.load(part_paths[part_i], mmap_mode="r")
        local_rows = np.array([p[1] for p in pairs], dtype=np.int64)
        chunk = np.array(mmap[local_rows], dtype=np.float32, copy=True)
        for (out_pos, _), vec in zip(pairs, chunk):
            out[out_pos] = vec

    return out


def normalize_l2_inplace(x: np.ndarray) -> None:
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    faiss.normalize_L2(x)


def maybe_load_sequences(fasta_path: Optional[str], wanted_ids: List[str]) -> Dict[str, str]:
    if fasta_path is None:
        return {}
    wanted = set(wanted_ids)
    found: Dict[str, str] = {}
    for sid, seq in iter_fasta(fasta_path):
        if sid in wanted:
            found[sid] = seq
            if len(found) == len(wanted):
                break
    return found


def main():
    ap = argparse.ArgumentParser(description="Test retrieval correctness for TMVec FAISS index")
    ap.add_argument("--index_file", required=True, help="Path to .index file")
    ap.add_argument(
        "--ids_file",
        default="",
        help="Optional concatenated IDs file for human-readable examples",
    )
    ap.add_argument("--embeddings_dir", required=True, help="Directory containing part_*.npy")
    ap.add_argument("--sample_size", type=int, default=2000, help="Number of vectors to test")
    ap.add_argument(
        "--sample_part",
        type=int,
        default=-1,
        help="If >=0, sample queries only from this part index (fast path; e.g. 0)",
    )
    ap.add_argument("--k", type=int, default=10, help="Top-k to retrieve")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--nprobe", type=int, default=0, help="Override nprobe for IVF indices")
    ap.add_argument(
        "--no_normalize_queries",
        action="store_true",
        help="Do not L2-normalize query vectors before search",
    )
    ap.add_argument(
        "--fasta",
        default="",
        help="Optional FASTA(.gz) to print sequence snippets for examples (requires --ids_file)",
    )
    ap.add_argument("--show_examples", type=int, default=5, help="How many query examples to print")
    args = ap.parse_args()

    print(f"[info] Loading index: {args.index_file}")
    index = faiss.read_index(args.index_file)
    if args.nprobe > 0 and hasattr(index, "nprobe"):
        index.nprobe = args.nprobe
    if hasattr(index, "nprobe"):
        print(f"[info] nprobe={index.nprobe}")
    print(f"[info] index.ntotal={index.ntotal}")

    all_ids: Optional[List[str]] = None
    if args.ids_file:
        print(f"[info] Loading IDs: {args.ids_file}")
        all_ids = read_ids_file(args.ids_file)
        if len(all_ids) != int(index.ntotal):
            raise RuntimeError(f"IDs count ({len(all_ids)}) != index.ntotal ({index.ntotal})")
    else:
        print("[info] No --ids_file provided; using FAISS row-index based validation")

    print(f"[info] Scanning embeddings: {args.embeddings_dir}")
    part_paths = sorted(glob(str(Path(args.embeddings_dir) / "part_*.npy")))
    if not part_paths:
        raise RuntimeError(f"No part_*.npy found in {args.embeddings_dir}")

    if args.sample_part >= 0:
        if args.sample_part >= len(part_paths):
            raise RuntimeError(f"--sample_part={args.sample_part} out of range [0, {len(part_paths)-1}]")
        cum_counts = build_partial_cum(part_paths, args.sample_part)
        total_vecs = int(index.ntotal)
    else:
        _, cum_counts = build_part_layout(args.embeddings_dir)
        total_vecs = cum_counts[-1]
        if total_vecs != int(index.ntotal):
            raise RuntimeError(f"Total vectors in parts ({total_vecs}) != index.ntotal ({index.ntotal})")
        if all_ids is not None and total_vecs != len(all_ids):
            raise RuntimeError(f"Total vectors in parts ({total_vecs}) != IDs count ({len(all_ids)})")

    n = min(args.sample_size, total_vecs)
    rng = np.random.default_rng(args.seed)
    if args.sample_part >= 0:
        part_i = args.sample_part
        part_n = int(np.load(part_paths[part_i], mmap_mode="r").shape[0])
        n = min(n, part_n)
        local = rng.choice(part_n, size=n, replace=False)
        base = 0 if part_i == 0 else cum_counts[part_i - 1]
        sample_idx = base + local
        print(f"[info] Sampling {n} queries from part_{part_i:05d}.npy")
    else:
        sample_idx = rng.choice(total_vecs, size=n, replace=False)
    true_rows = sample_idx.tolist()

    print(f"[info] Fetching {n} sampled vectors from shards")
    queries = fetch_vectors_by_global_idx(part_paths, cum_counts, sample_idx)
    if not args.no_normalize_queries:
        normalize_l2_inplace(queries)

    print(f"[info] Searching top-{args.k}")
    distances, neighbors = index.search(queries, args.k)

    hit_at_1 = 0
    hit_at_k = 0
    for i in range(n):
        retrieved_rows = []
        for j in range(args.k):
            row_idx = int(neighbors[i, j])
            if row_idx < 0:
                continue
            retrieved_rows.append(row_idx)
        if retrieved_rows and retrieved_rows[0] == true_rows[i]:
            hit_at_1 += 1
        if true_rows[i] in retrieved_rows:
            hit_at_k += 1

    r1 = hit_at_1 / n if n else 0.0
    rk = hit_at_k / n if n else 0.0
    print(f"[result] recall@1={r1:.4f} ({hit_at_1}/{n})")
    print(f"[result] recall@{args.k}={rk:.4f} ({hit_at_k}/{n})")

    show_n = min(args.show_examples, n)
    if show_n > 0:
        seq_map: Dict[str, str] = {}
        wanted: List[str] = []
        if args.fasta and all_ids is not None:
            wanted = list(dict.fromkeys([all_ids[r] for r in true_rows[:show_n]]))
            seq_map = maybe_load_sequences(args.fasta, wanted)
        print("[examples]")
        for i in range(show_n):
            qrow = true_rows[i]
            qid = all_ids[qrow] if all_ids is not None else f"row:{qrow}"
            top_i = int(neighbors[i, 0])
            top_id = all_ids[top_i] if (top_i >= 0 and all_ids is not None) else (f"row:{top_i}" if top_i >= 0 else "N/A")
            top_s = float(distances[i, 0]) if top_i >= 0 else float("nan")
            ok = "OK" if qid == top_id else "MISS"
            print(f"  {i+1}. query={qid} top1={top_id} top1_score={top_s:.6f} [{ok}]")
            if qid in seq_map:
                seq = seq_map[qid]
                print(f"     query_seq_prefix={seq[:80]}")


if __name__ == "__main__":
    main()
