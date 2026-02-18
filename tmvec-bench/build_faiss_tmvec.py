#!/usr/bin/env python3 -u
"""
Build a FAISS index from TMVec-2s embedding shards produced by embed_tmvec2s_from_tmvecbench.py.

Inputs (in --embeddings_dir):
  part_00000.npy        float32 array [N, 512]
  part_00000.ids.txt    N lines, same order as rows of .npy
  ...

Outputs:
  --index_file          FAISS index file (.index)
  --ids_file            concatenated mapping file (line i -> vector i)

Default behavior:
  - METRIC_INNER_PRODUCT + L2-normalize vectors => cosine similarity search.
"""

import argparse
import gzip
import logging
import os
import time
from glob import glob
from typing import List, Tuple, Optional

import numpy as np
import faiss

import torch

logger = logging.getLogger("tmvec_faiss")


def open_text(path: str, mode: str = "rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def normalize_L2_inplace(x: np.ndarray) -> None:
    """In-place L2 normalization of float32 matrix."""
    # faiss.normalize_L2 expects float32 contiguous
    if x.dtype != np.float32:
        x[:] = x.astype(np.float32, copy=False)
    if not x.flags["C_CONTIGUOUS"]:
        x[:] = np.ascontiguousarray(x)
    faiss.normalize_L2(x)


class FAISSIndexBuilder:
    """
    Like your run_faiss.py builder, but specialized for tmvec parts and streaming IDs.

    Supported:
      - FlatIP
      - IVFFlat
      - IVFPQ
      - IVFSQ
    """

    def __init__(
        self,
        embed_dim: int,
        index_type: str = "IVFPQ",
        nlist: int = 4096,
        pq_m: int = 64,
        pq_bits: int = 8,
        nprobe: int = 128,
        train_size: int = 256_000,
        use_gpu: bool = True,
        normalize: bool = True,
        add_chunk_size: int = 100_000,
        seed: int = 0,
    ):
        self.embed_dim = embed_dim
        self.index_type = index_type.upper()
        self.nlist = nlist
        self.pq_m = pq_m
        self.pq_bits = pq_bits
        self.nprobe = nprobe
        self.train_size = train_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.normalize = normalize
        self.add_chunk_size = add_chunk_size
        self.seed = seed

        self.gpu_resources = None
        self.index = None

        if self.index_type == "IVFPQ":
            if embed_dim % pq_m != 0:
                raise ValueError(f"pq_m={pq_m} must divide embed_dim={embed_dim} (TMVec-2s is usually 512).")

        logger.info(
            f"FAISSIndexBuilder(type={self.index_type}, dim={embed_dim}, nlist={nlist}, "
            f"pq_m={pq_m}, pq_bits={pq_bits}, nprobe={nprobe}, gpu={self.use_gpu}, "
            f"normalize={self.normalize}, train_size={train_size})"
        )

        np.random.seed(seed)

    def _set_gpu_config(self, config):
        config.device = torch.cuda.current_device()
        config.indicesOptions = faiss.INDICES_32_BIT
        config.useFloat16 = True
        return config

    def _create_gpu_index(self):
        self.gpu_resources = faiss.StandardGpuResources()
        d = self.embed_dim

        if self.index_type == "FLATIP":
            cfg = self._set_gpu_config(faiss.GpuIndexFlatConfig())
            return faiss.GpuIndexFlatIP(self.gpu_resources, d, cfg)

        if self.index_type == "IVFFLAT":
            cfg = self._set_gpu_config(faiss.GpuIndexIVFFlatConfig())
            return faiss.GpuIndexIVFFlat(
                self.gpu_resources, d, self.nlist, faiss.METRIC_INNER_PRODUCT, cfg
            )

        if self.index_type == "IVFPQ":
            cfg = self._set_gpu_config(faiss.GpuIndexIVFPQConfig())
            return faiss.GpuIndexIVFPQ(
                self.gpu_resources, d, self.nlist, self.pq_m, self.pq_bits,
                faiss.METRIC_INNER_PRODUCT, cfg
            )

        if self.index_type == "IVFSQ":
            cfg = self._set_gpu_config(faiss.GpuIndexIVFScalarQuantizerConfig())
            qtype = faiss.ScalarQuantizer.QT_fp16
            return faiss.GpuIndexIVFScalarQuantizer(
                self.gpu_resources, d, self.nlist, qtype,
                faiss.METRIC_INNER_PRODUCT, True, cfg
            )

        raise ValueError(f"Unsupported index_type={self.index_type}")

    def _create_cpu_index(self):
        d = self.embed_dim
        if self.index_type == "FLATIP":
            key = "Flat"
        elif self.index_type == "IVFFLAT":
            key = f"IVF{self.nlist},Flat"
        elif self.index_type == "IVFPQ":
            key = f"IVF{self.nlist},PQ{self.pq_m}x{self.pq_bits}"
        elif self.index_type == "IVFSQ":
            key = f"IVF{self.nlist},SQ8"
        else:
            raise ValueError(f"Unsupported index_type={self.index_type}")

        logger.info(f"Creating CPU index: index_factory({d}, '{key}', METRIC_INNER_PRODUCT)")
        return faiss.index_factory(d, key, faiss.METRIC_INNER_PRODUCT)

    def _collect_training_set(self, part_paths: List[str]) -> np.ndarray:
        """
        Proportional sampling across shards (same spirit as your run_faiss.py).
        Loads only the sampled rows into RAM.
        """
        logger.info(f"Collecting train set: target={self.train_size:,} from {len(part_paths)} shards")

        # Count total vectors
        total_vecs = 0
        shard_sizes = []
        for p in part_paths:
            mat = np.load(p, mmap_mode="r")
            n = int(mat.shape[0])
            shard_sizes.append(n)
            total_vecs += n

        if total_vecs == 0:
            raise RuntimeError("No vectors found in parts.")

        remaining = self.train_size
        samples = []

        for p, n in zip(part_paths, shard_sizes):
            if remaining <= 0:
                break

            # proportional sample, at least 1 per shard (when possible)
            n_sample = max(1, int(self.train_size * n / total_vecs))
            n_sample = min(n_sample, n, remaining)
            if n_sample <= 0:
                continue

            mat = np.load(p, mmap_mode="r")
            idx = np.random.choice(n, size=n_sample, replace=False)
            chunk = np.array(mat[idx], dtype=np.float32, copy=True)  # copy out of mmap
            if self.normalize:
                normalize_L2_inplace(chunk)
            samples.append(chunk)
            remaining -= n_sample

        train = np.concatenate(samples, axis=0)
        if train.shape[0] > self.train_size:
            train = train[: self.train_size]
        train = np.ascontiguousarray(train.astype(np.float32, copy=False))

        logger.info(f"Train set ready: {train.shape}")
        return train

    def _train_index(self, train_set: np.ndarray):
        logger.info("Training index...")
        t0 = time.time()

        if self.use_gpu:
            self.index = self._create_gpu_index()
            self.index.train(train_set)  # numpy is fine; FAISS copies to GPU
        else:
            self.index = self._create_cpu_index()
            self.index.train(train_set)

        logger.info(f"Index trained in {time.time() - t0:.1f}s")

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

    def _add_part(
        self,
        part_npy: str,
        part_ids: str,
        ids_out_fh,
        start_row: int = 0,
    ) -> int:
        """
        Add one shard (.npy) and write its IDs to ids_out_fh in the same order.
        Supports resuming within a shard via start_row.
        Returns number of vectors added.
        """
        mat_mmap = np.load(part_npy, mmap_mode="r")
        n_total = int(mat_mmap.shape[0])

        if start_row >= n_total:
            return 0

        # Open ids file and skip start_row lines if resuming mid-shard
        with open_text(part_ids, "rt") as id_fh:
            for _ in range(start_row):
                if not id_fh.readline():
                    raise RuntimeError(f"IDs file ended early while skipping: {part_ids}")

            added = 0
            for s in range(start_row, n_total, self.add_chunk_size):
                e = min(s + self.add_chunk_size, n_total)
                chunk = np.array(mat_mmap[s:e], dtype=np.float32, copy=True)

                if self.normalize:
                    normalize_L2_inplace(chunk)

                if self.use_gpu:
                    self.index.add(chunk)
                else:
                    self.index.add(chunk)

                # Write matching IDs
                need = e - s
                for _ in range(need):
                    line = id_fh.readline()
                    if not line:
                        raise RuntimeError(f"IDs file ended early: {part_ids} (needed {need} more)")
                    ids_out_fh.write(line if line.endswith("\n") else (line + "\n"))

                added += need

        return added

    def save_index(self, path: str):
        assert self.index is not None, "No index to save"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if self.use_gpu:
            logger.info("Converting GPU index to CPU for saving...")
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index

        faiss.write_index(cpu_index, path)
        logger.info(f"Saved index: {path} (ntotal={cpu_index.ntotal:,})")

    def build_from_parts(
        self,
        parts_dir: str,
        index_file: str,
        ids_file: str,
        resume: bool = False,
        save_every_parts: int = 0,
    ):
        # Discover parts
        part_paths = sorted(glob(os.path.join(parts_dir, "part_*.npy")))
        if not part_paths:
            raise RuntimeError(f"No part_*.npy found in {parts_dir}")

        # Verify ids files exist
        id_paths = []
        for p in part_paths:
            ids = p.replace(".npy", ".ids.txt")
            if not os.path.exists(ids) and os.path.exists(ids + ".gz"):
                ids = ids + ".gz"
            if not os.path.exists(ids):
                raise RuntimeError(f"Missing IDs file for {p}: expected {p.replace('.npy', '.ids.txt')}[.gz]")
            id_paths.append(ids)

        # Resume support (optional): requires existing index file and ids file.
        start_part_idx = 0
        start_row_in_part = 0

        if resume and os.path.exists(index_file):
            logger.info(f"[resume] loading existing index: {index_file}")
            idx = faiss.read_index(index_file)
            already = int(idx.ntotal)
            logger.info(f"[resume] index has ntotal={already:,}")

            # Create new index object in this builder
            if self.use_gpu:
                # Move to GPU
                res = faiss.StandardGpuResources()
                opts = faiss.GpuClonerOptions()
                opts.useFloat16 = True
                opts.indicesOptions = faiss.INDICES_32_BIT
                self.index = faiss.index_cpu_to_gpu(res, torch.cuda.current_device(), idx, opts)
            else:
                self.index = idx

            if hasattr(self.index, "nprobe"):
                self.index.nprobe = self.nprobe

            # Determine which part/row to resume from based on cumulative sizes
            cum = 0
            for i, p in enumerate(part_paths):
                n = int(np.load(p, mmap_mode="r").shape[0])
                if cum + n > already:
                    start_part_idx = i
                    start_row_in_part = already - cum
                    break
                cum += n
            else:
                logger.info("[resume] all parts already indexed; nothing to do.")
                return

            # Append to ids_file (do not overwrite)
            ids_mode = "at"
        else:
            # Fresh build: train then add from scratch
            if self.index_type != "FLATIP":
                train = self._collect_training_set(part_paths)
                self._train_index(train)
                del train
            else:
                self.index = self._create_gpu_index() if self.use_gpu else self._create_cpu_index()
                if hasattr(self.index, "nprobe"):
                    self.index.nprobe = self.nprobe
            ids_mode = "wt"

        logger.info(f"Writing IDs to: {ids_file} (mode={ids_mode})")
        os.makedirs(os.path.dirname(ids_file) or ".", exist_ok=True)
        with open_text(ids_file, ids_mode) as ids_out:
            total_added = 0
            for i in range(start_part_idx, len(part_paths)):
                part_npy = part_paths[i]
                part_ids = id_paths[i]
                row0 = start_row_in_part if i == start_part_idx else 0

                logger.info(f"Adding {os.path.basename(part_npy)} (start_row={row0})")
                added = self._add_part(part_npy, part_ids, ids_out, start_row=row0)
                total_added += added
                logger.info(f"  added {added:,} vectors (index.ntotal={self.index.ntotal:,})")

                # after first resumed part, reset row offset
                start_row_in_part = 0

                if save_every_parts and (i + 1) % save_every_parts == 0:
                    tmp = index_file + ".partial"
                    self.save_index(tmp)
                    os.replace(tmp, index_file)
                    logger.info(f"  snapshot saved: {index_file}")

        # Final save
        self.save_index(index_file)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description="Build FAISS index from TMVec-2s part_*.npy shards")
    ap.add_argument("--embeddings_dir", required=True, help="Directory with part_*.npy and part_*.ids.txt")
    ap.add_argument("--index_file", required=True, help="Output FAISS index file (.index)")
    ap.add_argument("--ids_file", default="", help="Output IDs mapping file (txt or txt.gz). Default: <index_file>_ids.txt.gz")

    ap.add_argument("--index_type", default="IVFPQ", choices=["FlatIP", "IVFFlat", "IVFPQ", "IVFSQ"])
    ap.add_argument("--nlist", type=int, default=65536)
    ap.add_argument("--pq_m", type=int, default=64)
    ap.add_argument("--pq_bits", type=int, default=8)
    ap.add_argument("--nprobe", type=int, default=128)
    ap.add_argument("--train_size", type=int, default=500_000)

    ap.add_argument("--nogpu", action="store_true")
    ap.add_argument("--no_normalize", action="store_true", help="Disable L2 normalization (NOT recommended for cosine/IP)")

    ap.add_argument("--add_chunk_size", type=int, default=100_000, help="Vectors per add() call")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--resume", action="store_true", help="Resume if --index_file exists (appends to ids_file)")
    ap.add_argument("--save_every_parts", type=int, default=0, help="Save index snapshot every N parts (enables resume safety)")

    args = ap.parse_args()

    ids_file = args.ids_file
    if not ids_file:
        base = args.index_file
        if base.endswith(".index"):
            base = base[:-6]
        ids_file = base + "_ids.txt.gz"

    # Infer dim from first part
    first_parts = sorted(glob(os.path.join(args.embeddings_dir, "part_*.npy")))
    if not first_parts:
        raise SystemExit(f"No part_*.npy found in {args.embeddings_dir}")
    d = int(np.load(first_parts[0], mmap_mode="r").shape[1])
    logger.info(f"Inferred embedding dim={d} from {os.path.basename(first_parts[0])}")

    builder = FAISSIndexBuilder(
        embed_dim=d,
        index_type=args.index_type,
        nlist=args.nlist,
        pq_m=args.pq_m,
        pq_bits=args.pq_bits,
        nprobe=args.nprobe,
        train_size=args.train_size,
        use_gpu=(not args.nogpu),
        normalize=(not args.no_normalize),
        add_chunk_size=args.add_chunk_size,
        seed=args.seed,
    )

    builder.build_from_parts(
        parts_dir=args.embeddings_dir,
        index_file=args.index_file,
        ids_file=ids_file,
        resume=args.resume,
        save_every_parts=args.save_every_parts,
    )


if __name__ == "__main__":
    main()
