#!/usr/bin/env python3 -u
"""
Build a FAISS index for protein sequences using ESM-2 mean-pooled embeddings.

Features:
- Multi-GPU data-parallel embedding (scales linearly with GPU count)
- Shard-level resume: detects completed shards and skips them on restart
- Streaming or two-step (embed → index) workflow
- Support for FlatIP, IVFFlat, IVFPQ, IVFSQ index types
- Handles UniRef50-scale databases (~60M sequences)

Multi-GPU usage:
    # Automatic: uses all visible GPUs
    python build_faiss_index.py esm2_t33_650M_UR50D uniref50.fasta uniref50.index \\
        --embed_only --embeddings_dir ./embeds/

    # Control GPU count:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python build_faiss_index.py ...

    # Or with torchrun for multi-node:
    torchrun --nproc_per_node=4 build_faiss_index.py ... --distributed

Single-GPU / CPU:
    python build_faiss_index.py esm2_t33_650M_UR50D pfam.fasta pfam.index \\
        --index_type FlatIP --streaming

Two-step with resume:
    # Step 1 (can kill and restart — skips completed shards):
    python build_faiss_index.py esm2_t33_650M_UR50D uniref50.fasta uniref50.index \\
        --embed_only --embeddings_dir ./embeds/

    # Step 2:
    python build_faiss_index.py esm2_t33_650M_UR50D uniref50.fasta uniref50.index \\
        --index_only --embeddings_dir ./embeds/ \\
        --index_type IVFPQ --nlist 65536 --pq_m 32 --train_size 5000000
"""
import argparse
import json
import logging
import os
import pathlib
import pickle
import sys
import time
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from esm import pretrained, FastaBatchedDataset

from esm2 import ESM2

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────
FAISSGPUIndex = Union[
    faiss.GpuIndexIVFFlat,
    faiss.GpuIndexIVFPQ,
    faiss.GpuIndexIVFScalarQuantizer,
    faiss.GpuIndexFlatIP,
]


# ──────────────────────────────────────────────────────────────
# FASTA Chunking (for multi-GPU assignment)
# ──────────────────────────────────────────────────────────────

def read_fasta_sequences(fasta_path: str) -> List[Tuple[str, str]]:
    """
    Read all (header, sequence) pairs from a FASTA file.
    Returns list of (label, sequence) tuples.
    """
    entries = []
    header = None
    seq_parts = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seq_parts)))
                header = line[1:].split()[0]
                seq_parts = []
            elif header is not None:
                seq_parts.append(line)
    if header is not None:
        entries.append((header, "".join(seq_parts)))

    return entries


def chunk_sequences(
    entries: List[Tuple[str, str]],
    shard_size: int,
) -> List[List[Tuple[str, str]]]:
    """Split sequence list into chunks of shard_size."""
    chunks = []
    for i in range(0, len(entries), shard_size):
        chunks.append(entries[i : i + shard_size])
    return chunks


# ──────────────────────────────────────────────────────────────
# ESM-2 Embedding with Multi-GPU Support
# ──────────────────────────────────────────────────────────────

class ESMEmbedder:
    """
    Embeds protein sequences using ESM-2 with multi-GPU DataParallel.

    Multi-GPU strategy:
        - Load ESM-2, wrap in nn.DataParallel across all available GPUs
        - Each forward pass distributes the batch across GPUs automatically
        - Shard-level checkpointing: completed shards are detected on resume

    This is simpler and more robust than torch.distributed for this use case
    because we're doing pure inference with no gradient communication.
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        repr_layer: int = 33,
        toks_per_batch: int = 4096,
        truncation_seq_length: int = 1022,
        use_gpu: bool = True,
        gpu_ids: Optional[List[int]] = None,
    ):
        self.model_name = model_name
        self.repr_layer = repr_layer
        self.toks_per_batch = toks_per_batch
        self.truncation_seq_length = truncation_seq_length

        logger.info(f"Loading ESM-2 model: {model_name}")
        model, self.alphabet = pretrained.load_model_and_alphabet(model_name)
        self.model = ESM2(alphabet=self.alphabet, 
                          num_layers=model.num_layers, 
                          embed_dim=model.embed_dim, 
                          attention_heads=model.attention_heads)
        state_dict = model.state_dict()
        state_dict_packed = self.model.upgrade_state_dict_qkv_to_packed(state_dict.copy())
        self.model.load_state_dict(state_dict_packed,strict=False)
        self.model.eval()
        del model

        # ── GPU setup ──
        self.device = torch.device("cpu")
        self.num_gpus = 0

        if use_gpu and torch.cuda.is_available():
            available_gpus = list(range(torch.cuda.device_count()))
            if gpu_ids is not None:
                available_gpus = [g for g in gpu_ids if g < torch.cuda.device_count()]

            self.num_gpus = len(available_gpus)

            if self.num_gpus > 1:
                logger.info(f"Using DataParallel across {self.num_gpus} GPUs: {available_gpus}")
                self.model = self.model.cuda(available_gpus[0])
                self.model = nn.DataParallel(self.model, device_ids=available_gpus)
                self.device = torch.device(f"cuda:{available_gpus[0]}")
            elif self.num_gpus == 1:
                self.model = self.model.cuda(available_gpus[0])
                self.device = torch.device(f"cuda:{available_gpus[0]}")
                logger.info(f"Using single GPU: {available_gpus[0]}")
            else:
                logger.warning("No usable GPUs found, falling back to CPU")
        else:
            logger.info("Running on CPU")

        # Get embed dim from unwrapped model
        base_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        self.embed_dim = base_model.embed_tokens.embedding_dim
        self.num_layers = base_model.num_layers
        logger.info(f"Embedding dimension: {self.embed_dim}, layers: {self.num_layers}")

    def _mean_pool(
        self,
        representations: torch.Tensor,
        tokens: torch.Tensor,
    ) -> np.ndarray:
        """
        Masked mean pooling: average over valid tokens,
        excluding <cls>, <pad>, and <eos>.
        """
        mask = (
            (tokens != self.alphabet.padding_idx)
            & (tokens != self.alphabet.cls_idx)
            & (tokens != self.alphabet.eos_idx)
        )
        mask_expanded = mask.unsqueeze(-1).expand(representations.size()).float()
        sum_embeddings = torch.sum(representations * mask_expanded, dim=1)
        count_valid = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / count_valid
        return mean_pooled.cpu().numpy().astype(np.float32)

    def create_dataloader(self, fasta_path: str):
        """Create a batched dataloader from a FASTA file."""
        dataset = FastaBatchedDataset.from_file(fasta_path)
        batches = dataset.get_batch_indices(
            self.toks_per_batch, extra_toks_per_seq=1
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=self.alphabet.get_batch_converter(self.truncation_seq_length),
            batch_sampler=batches,
        )
        logger.info(
            f"Created dataloader: {len(dataset)} sequences, {len(batches)} batches"
        )
        return data_loader, dataset

    def create_dataloader_from_entries(
        self,
        entries: List[Tuple[str, str]],
    ):
        """
        Create a dataloader from a pre-loaded list of (label, sequence) tuples.
        Uses ESM's FastaBatchedDataset internals for consistent batching.
        """
        # FastaBatchedDataset stores .sequence_labels and .sequence_strs
        dataset = FastaBatchedDataset(
            sequence_labels=[e[0] for e in entries],
            sequence_strs=[e[1] for e in entries],
        )
        batches = dataset.get_batch_indices(
            self.toks_per_batch, extra_toks_per_seq=1
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=self.alphabet.get_batch_converter(self.truncation_seq_length),
            batch_sampler=batches,
        )
        return data_loader

    @torch.no_grad()
    def embed_batches(self, data_loader, max_batches: Optional[int] = None):
        """
        Generator that yields (labels, mean_pooled_embeddings) per batch.
        Works with both single-GPU and DataParallel models.
        """
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            toks = toks.to(device=self.device, non_blocking=True)

            # DataParallel returns the same dict structure, but we need
            # to handle the repr_layers argument carefully
            out = self.model(toks, repr_layers=[self.repr_layer])
            reprs = out["representations"][self.repr_layer]

            # reprs may be on different device if DataParallel gathered to GPU 0
            # _mean_pool handles .cpu() internally
            mean_pooled = self._mean_pool(reprs, toks)

            yield labels, mean_pooled

    @torch.no_grad()
    def embed_fasta_to_shards(
        self,
        fasta_path: str,
        output_dir: str,
        shard_size: int = 100_000,
    ) -> Tuple[int, List[str]]:
        """
        Embed all sequences and save as sharded .npy/.pkl files.
        Supports resume: detects and skips completed shards.

        Strategy:
        1. Read full FASTA into memory (just headers + sequences, not embeddings)
        2. Pre-assign sequences to shards deterministically
        3. For each shard: check if .npy already exists → skip if so
        4. Embed remaining shards, potentially across multiple GPUs

        This ensures that shard boundaries are stable across runs,
        so resume always produces the same output.
        """
        os.makedirs(output_dir, exist_ok=True)

        # ── Step 1: Read all sequences and assign to shards ──
        logger.info(f"Reading FASTA: {fasta_path}")
        all_entries = read_fasta_sequences(fasta_path)
        total_seqs = len(all_entries)
        logger.info(f"Total sequences: {total_seqs:,}")

        chunks = chunk_sequences(all_entries, shard_size)
        total_shards = len(chunks)
        logger.info(f"Total shards ({shard_size} seqs each): {total_shards}")

        # ── Step 2: Detect completed shards ──
        completed_shards = set()
        for shard_idx in range(total_shards):
            npy_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.npy")
            pkl_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.pkl")
            # A shard is complete only if both .npy and .pkl exist and .npy has right shape
            if os.path.exists(npy_path) and os.path.exists(pkl_path):
                try:
                    mat = np.load(npy_path, mmap_mode="r")
                    expected_size = len(chunks[shard_idx])
                    if mat.shape[0] == expected_size and mat.shape[1] == self.embed_dim:
                        completed_shards.add(shard_idx)
                except Exception:
                    pass  # corrupted file, will re-do

        pending_shards = [i for i in range(total_shards) if i not in completed_shards]
        logger.info(
            f"Resume status: {len(completed_shards)} complete, "
            f"{len(pending_shards)} pending"
        )

        if not pending_shards:
            logger.info("All shards already complete!")
            shard_paths = [
                os.path.join(output_dir, f"shard_{i:05d}.npy")
                for i in range(total_shards)
            ]
            self._save_metadata(output_dir, total_seqs, total_shards)
            return total_seqs, shard_paths

        # ── Step 3: Embed pending shards ──
        logger.info(f"Embedding {len(pending_shards)} remaining shards...")
        t0 = time.time()

        for progress_idx, shard_idx in enumerate(pending_shards):
            chunk_entries = chunks[shard_idx]
            labels = [e[0] for e in chunk_entries]
            sequences = [e[1] for e in chunk_entries]

            # Create a dataloader for just this chunk
            data_loader = self.create_dataloader_from_entries(chunk_entries)

            # Embed all batches in this chunk
            shard_embeddings = []
            shard_labels = []

            for batch_labels, batch_embeddings in self.embed_batches(data_loader):
                shard_embeddings.append(batch_embeddings)
                shard_labels.extend(batch_labels)

            # Concatenate and save
            mat = np.concatenate(shard_embeddings, axis=0).astype(np.float32)

            # Atomic save: write to temp file then rename (prevents partial shards)
            npy_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.npy")
            pkl_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.pkl")
            tmp_npy = npy_path + ".tmp"
            tmp_pkl = pkl_path + ".tmp"

            np.save(tmp_npy, mat)
            with open(tmp_npy, "rb") as f:
                os.fsync(f.fileno())

            with open(tmp_pkl, "wb") as f:
                pickle.dump(shard_labels, f)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_npy, npy_path)
            os.replace(tmp_pkl, pkl_path)

            elapsed = time.time() - t0
            rate = (progress_idx + 1) / elapsed * 3600 if elapsed > 0 else 0
            eta_hours = (len(pending_shards) - progress_idx - 1) / rate * 3600 if rate > 0 else 0

            logger.info(
                f"  Shard {shard_idx:5d}/{total_shards} "
                f"({progress_idx + 1}/{len(pending_shards)} pending) | "
                f"{mat.shape[0]:,} vectors | "
                f"Rate: {rate:.0f} shards/hr | "
                f"ETA: {eta_hours / 3600:.1f}h"
            )

        # ── Step 4: Save metadata ──
        self._save_metadata(output_dir, total_seqs, total_shards)

        shard_paths = [
            os.path.join(output_dir, f"shard_{i:05d}.npy")
            for i in range(total_shards)
        ]

        elapsed_total = time.time() - t0
        logger.info(
            f"Embedding complete: {total_seqs:,} sequences, "
            f"{total_shards} shards, {elapsed_total / 3600:.1f}h total"
        )

        return total_seqs, shard_paths

    def _save_metadata(self, output_dir: str, total_seqs: int, num_shards: int):
        """Save embedding metadata for the index-building step."""
        meta = {
            "total_sequences": total_seqs,
            "embed_dim": self.embed_dim,
            "model": self.model_name,
            "repr_layer": self.repr_layer,
            "num_shards": num_shards,
            "num_gpus_used": self.num_gpus,
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)


# ──────────────────────────────────────────────────────────────
# Multi-GPU Embedding via torchrun / torch.distributed
# (Alternative to DataParallel for multi-node setups)
# ──────────────────────────────────────────────────────────────

def run_distributed_embedding(args):
    """
    Distributed embedding using torch.distributed.
    Each rank processes a non-overlapping subset of shards.

    Launch with:
        torchrun --nproc_per_node=NUM_GPUS build_faiss_index.py ... --distributed
    """
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)

    # Setup logging only for rank 0
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    logger.info(f"Rank {rank}/{world_size}, local_rank={local_rank}")

    output_dir = args.embeddings_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Read FASTA and chunk into shards (all ranks do this) ──
    all_entries = read_fasta_sequences(str(args.fasta_file))
    chunks = chunk_sequences(all_entries, args.shard_size)
    total_shards = len(chunks)

    if rank == 0:
        logger.info(f"Total: {len(all_entries):,} sequences, {total_shards} shards")

    # ── Assign shards to this rank (round-robin) ──
    my_shard_indices = list(range(rank, total_shards, world_size))
    logger.info(f"Rank {rank}: assigned {len(my_shard_indices)} shards")

    # ── Create single-GPU embedder for this rank ──
    embedder = ESMEmbedder(
        model_name=args.model_location,
        repr_layer=args.repr_layer,
        toks_per_batch=args.toks_per_batch,
        truncation_seq_length=args.truncation_seq_length,
        use_gpu=True,
        gpu_ids=[local_rank],
    )

    # ── Process assigned shards with resume ──
    for progress_idx, shard_idx in enumerate(my_shard_indices):
        npy_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.npy")
        pkl_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.pkl")

        # Resume: skip completed shards
        if os.path.exists(npy_path) and os.path.exists(pkl_path):
            try:
                mat = np.load(npy_path, mmap_mode="r")
                if mat.shape[0] == len(chunks[shard_idx]) and mat.shape[1] == embedder.embed_dim:
                    if rank == 0 and progress_idx % 10 == 0:
                        logger.info(f"  Shard {shard_idx} already complete, skipping")
                    continue
            except Exception:
                pass

        chunk_entries = chunks[shard_idx]
        data_loader = embedder.create_dataloader_from_entries(chunk_entries)

        shard_embeddings = []
        shard_labels = []
        for batch_labels, batch_embeddings in embedder.embed_batches(data_loader):
            shard_embeddings.append(batch_embeddings)
            shard_labels.extend(batch_labels)

        mat = np.concatenate(shard_embeddings, axis=0).astype(np.float32)

        # Atomic save
        tmp_npy = npy_path + f".tmp.rank{rank}"
        tmp_pkl = pkl_path + f".tmp.rank{rank}"
        np.save(tmp_npy, mat)
        with open(tmp_pkl, "wb") as f:
            pickle.dump(shard_labels, f)
        os.replace(tmp_npy, npy_path)
        os.replace(tmp_pkl, pkl_path)

        if rank == 0 or progress_idx % 5 == 0:
            logger.info(
                f"  Rank {rank}: shard {shard_idx}/{total_shards} "
                f"({progress_idx + 1}/{len(my_shard_indices)}) | "
                f"{mat.shape[0]:,} vectors"
            )

    # ── Sync all ranks ──
    dist.barrier()

    if rank == 0:
        # Save metadata
        meta = {
            "total_sequences": len(all_entries),
            "embed_dim": embedder.embed_dim,
            "model": args.model_location,
            "repr_layer": args.repr_layer,
            "num_shards": total_shards,
            "num_gpus_used": world_size,
            "mode": "distributed",
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"All ranks complete. {total_shards} shards in {output_dir}")

    dist.destroy_process_group()


# ──────────────────────────────────────────────────────────────
# FAISS Index Builder (unchanged from original)
# ──────────────────────────────────────────────────────────────

class FAISSIndexBuilder:
    """
    Builds a FAISS index from embedding shards.

    Supports:
        - FlatIP: exact inner product search (small databases)
        - IVFFlat: inverted file with uncompressed vectors
        - IVFPQ: inverted file with product quantization (large databases)
        - IVFSQ: inverted file with scalar quantization

    Can build on GPU and save as CPU index for portability.
    """

    def __init__(
        self,
        embed_dim: int,
        index_type: str = "IVFPQ",
        nlist: int = 4096,
        pq_m: int = 32,
        pq_bits: int = 8,
        nprobe: int = 128,
        use_gpu: bool = True,
        train_size: int = 256_000,
    ):
        self.embed_dim = embed_dim
        self.index_type = index_type.upper()
        self.nlist = nlist
        self.pq_m = pq_m
        self.pq_bits = pq_bits
        self.nprobe = nprobe
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.train_size = train_size

        self.gpu_resources = None
        self.index = None

        logger.info(
            f"FAISSIndexBuilder: type={self.index_type}, dim={embed_dim}, "
            f"nlist={nlist}, pq_m={pq_m}, gpu={self.use_gpu}"
        )

    def _set_gpu_config(self, config):
        """Set common GPU config options."""
        config.device = torch.cuda.current_device()
        config.indicesOptions = faiss.INDICES_32_BIT
        config.useFloat16 = True
        return config

    def _create_gpu_index(self) -> FAISSGPUIndex:
        """Create a GPU FAISS index based on the selected type."""
        self.gpu_resources = faiss.StandardGpuResources()
        d = self.embed_dim

        if self.index_type == "FLATIP":
            config = self._set_gpu_config(faiss.GpuIndexFlatConfig())
            return faiss.GpuIndexFlatIP(self.gpu_resources, d, config)

        elif self.index_type == "IVFFLAT":
            config = self._set_gpu_config(faiss.GpuIndexIVFFlatConfig())
            return faiss.GpuIndexIVFFlat(
                self.gpu_resources, d, self.nlist,
                faiss.METRIC_INNER_PRODUCT, config,
            )

        elif self.index_type == "IVFPQ":
            config = self._set_gpu_config(faiss.GpuIndexIVFPQConfig())
            return faiss.GpuIndexIVFPQ(
                self.gpu_resources, d, self.nlist,
                self.pq_m, self.pq_bits,
                faiss.METRIC_INNER_PRODUCT, config,
            )

        elif self.index_type == "IVFSQ":
            config = self._set_gpu_config(faiss.GpuIndexIVFScalarQuantizerConfig())
            qtype = faiss.ScalarQuantizer.QT_fp16
            return faiss.GpuIndexIVFScalarQuantizer(
                self.gpu_resources, d, self.nlist,
                qtype, faiss.METRIC_INNER_PRODUCT, True, config,
            )

        else:
            raise ValueError(
                f"Unsupported index type: {self.index_type}. "
                f"Choose from: FlatIP, IVFFlat, IVFPQ, IVFSQ"
            )

    def _create_cpu_index(self) -> faiss.Index:
        """Create a CPU FAISS index using the index factory."""
        d = self.embed_dim

        if self.index_type == "FLATIP":
            index_key = "Flat"
        elif self.index_type == "IVFFLAT":
            index_key = f"IVF{self.nlist},Flat"
        elif self.index_type == "IVFPQ":
            index_key = f"IVF{self.nlist},PQ{self.pq_m}x{self.pq_bits}"
        elif self.index_type == "IVFSQ":
            index_key = f"IVF{self.nlist},SQ8"
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        logger.info(f"CPU index: faiss.index_factory({d}, '{index_key}', METRIC_INNER_PRODUCT)")
        return faiss.index_factory(d, index_key, faiss.METRIC_INNER_PRODUCT)

    def _collect_training_set(self, shard_paths: List[str]) -> np.ndarray:
        """Collect a training set by proportional sampling from shards."""
        logger.info(f"Collecting training set: {self.train_size} vectors from {len(shard_paths)} shards")

        all_vectors = []
        remaining = self.train_size

        # Count total vectors
        total_vecs = 0
        for path in shard_paths:
            mat = np.load(path, mmap_mode="r")
            total_vecs += mat.shape[0]

        # Proportionally sample from each shard
        for path in shard_paths:
            mat = np.load(path, mmap_mode="r")
            n = mat.shape[0]
            n_sample = max(1, int(self.train_size * n / total_vecs))
            n_sample = min(n_sample, n, remaining)

            if n_sample <= 0:
                continue

            indices = np.random.choice(n, size=n_sample, replace=False)
            sampled = mat[indices].astype(np.float32)
            all_vectors.append(sampled)
            remaining -= n_sample

            if remaining <= 0:
                break

        train_set = np.concatenate(all_vectors, axis=0)[: self.train_size]
        logger.info(f"Training set shape: {train_set.shape}")
        return train_set.astype(np.float32)

    def _train_index(self, train_set: np.ndarray):
        """Train the FAISS index."""
        logger.info("Training FAISS index...")
        t0 = time.time()

        if self.use_gpu:
            self.index = self._create_gpu_index()
            self.index.train(np.ascontiguousarray(train_set.astype(np.float32, copy=False)))
        else:
            self.index = self._create_cpu_index()
            self.index.train(np.ascontiguousarray(train_set.astype(np.float32, copy=False)))

        elapsed = time.time() - t0
        logger.info(f"Index trained in {elapsed:.1f}s")

    def _add_vectors(self, shard_paths: List[str]):
        """Stream vectors from shards into the trained index."""
        logger.info(f"Adding vectors from {len(shard_paths)} shards...")
        total_added = 0

        for shard_idx, path in enumerate(tqdm(shard_paths, desc="Adding shards")):
            mat = np.load(path).astype(np.float32)

            if self.use_gpu:
                chunk_size = 100_000
                for start in range(0, mat.shape[0], chunk_size):
                    end = min(start + chunk_size, mat.shape[0])
                    chunk = np.ascontiguousarray(mat[start:end].astype(np.float32, copy=False))
                    self.index.add(chunk)
            else:
                self.index.add(np.ascontiguousarray(mat.astype(np.float32, copy=False)))

            total_added += mat.shape[0]

            if (shard_idx + 1) % 10 == 0:
                logger.info(f"  Added {total_added:,} vectors so far")

        logger.info(f"Total vectors in index: {self.index.ntotal:,}")

    def build_from_shards(self, shard_paths: List[str], save_path: str):
        """Full pipeline: collect training set → train → add vectors → save."""
        if self.index_type != "FLATIP":
            train_set = self._collect_training_set(shard_paths)
            self._train_index(train_set)
            del train_set
        else:
            if self.use_gpu:
                self.index = self._create_gpu_index()
            else:
                self.index = self._create_cpu_index()

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

        self._add_vectors(shard_paths)
        self.save_index(save_path)

    def build_streaming(
        self,
        embedder: ESMEmbedder,
        fasta_path: str,
        save_path: str,
    ):
        """
        One-pass pipeline: embed + build index without saving shards.
        Only uses single GPU (no multi-GPU for streaming mode).
        """
        data_loader, dataset = embedder.create_dataloader(fasta_path)

        # Phase 1: Collect training set
        if self.index_type != "FLATIP":
            logger.info(f"Phase 1: Sampling {self.train_size} vectors for training...")
            train_buffer = []
            train_count = 0

            for labels, embeddings in embedder.embed_batches(data_loader):
                train_buffer.append(embeddings)
                train_count += embeddings.shape[0]
                if train_count >= self.train_size:
                    break

            train_set = np.concatenate(train_buffer)[: self.train_size].astype(np.float32)
            del train_buffer
            self._train_index(train_set)
            del train_set
        else:
            if self.use_gpu:
                self.index = self._create_gpu_index()
            else:
                self.index = self._create_cpu_index()

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

        # Phase 2: Stream all vectors
        data_loader, _ = embedder.create_dataloader(fasta_path)
        logger.info("Phase 2: Streaming all sequences into the index...")

        all_labels = []
        total_added = 0

        for labels, embeddings in tqdm(
            embedder.embed_batches(data_loader), desc="Indexing", total=len(data_loader)
        ):
            if self.use_gpu:
                self.index.add(np.ascontiguousarray(embeddings.astype(np.float32, copy=False)))
            else:
                self.index.add(np.ascontiguousarray(embeddings.astype(np.float32, copy=False)))

            all_labels.extend(labels)
            total_added += embeddings.shape[0]

        logger.info(f"Total vectors in index: {self.index.ntotal:,}")

        self.save_index(save_path)

        labels_path = save_path.replace(".index", "").replace(".faiss", "") + "_labels.pkl"
        with open(labels_path, "wb") as f:
            pickle.dump(all_labels, f)
        logger.info(f"Labels saved to {labels_path}")

    def save_index(self, path: str):
        """Save the FAISS index (CPU format)."""
        assert self.index is not None, "No index to save"

        if self.use_gpu:
            logger.info("Converting GPU index to CPU for saving...")
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(cpu_index, str(path))
        logger.info(f"Index saved to {path} ({self.index.ntotal:,} vectors)")

    @staticmethod
    def load_index(path: str, use_gpu: bool = False, nprobe: int = 128) -> faiss.Index:
        """Load a FAISS index, optionally moving to GPU."""
        index = faiss.read_index(str(path))
        logger.info(f"Loaded index: {index.ntotal:,} vectors")

        if hasattr(index, "nprobe"):
            index.nprobe = nprobe

        if use_gpu and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            cloner_opts = faiss.GpuClonerOptions()
            cloner_opts.useFloat16 = True
            cloner_opts.usePrecomputed = False
            cloner_opts.indicesOptions = faiss.INDICES_32_BIT
            index = faiss.index_cpu_to_gpu(
                res, torch.cuda.current_device(), index, cloner_opts
            )
            logger.info("Index moved to GPU")

        return index


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def create_parser():
    parser = argparse.ArgumentParser(
        description="Build a FAISS index for protein sequences using ESM-2 embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-GPU embedding (4 GPUs, DataParallel, with resume):
  CUDA_VISIBLE_DEVICES=0,1,2,3 python build_faiss_index.py \\
      esm2_t33_650M_UR50D uniref50.fasta uniref50.index \\
      --embed_only --embeddings_dir ./embeds/

  # Multi-GPU embedding (torchrun, for multi-node):
  torchrun --nproc_per_node=4 build_faiss_index.py \\
      esm2_t33_650M_UR50D uniref50.fasta uniref50.index \\
      --embed_only --embeddings_dir ./embeds/ --distributed

  # Build index from saved shards:
  python build_faiss_index.py esm2_t33_650M_UR50D uniref50.fasta uniref50.index \\
      --index_only --embeddings_dir ./embeds/ \\
      --index_type IVFPQ --nlist 65536 --pq_m 32 --train_size 5000000

  # Small database, single pass:
  python build_faiss_index.py esm2_t33_650M_UR50D pfam.fasta pfam.index \\
      --streaming --index_type FlatIP

  # Resume interrupted embedding (just re-run the same command):
  CUDA_VISIBLE_DEVICES=0,1,2,3 python build_faiss_index.py \\
      esm2_t33_650M_UR50D uniref50.fasta uniref50.index \\
      --embed_only --embeddings_dir ./embeds/
        """,
    )

    # Positional
    parser.add_argument(
        "model_location", type=str,
        help="ESM-2 model name (e.g., esm2_t33_650M_UR50D, esm2_t36_3B_UR50D)",
    )
    parser.add_argument(
        "fasta_file", type=pathlib.Path,
        help="Input FASTA file",
    )
    parser.add_argument(
        "index_file", type=pathlib.Path,
        help="Output FAISS index path",
    )

    # ESM-2 options
    emb_group = parser.add_argument_group("ESM-2 embedding options")
    emb_group.add_argument("--repr_layer", type=int, default=12,
                           help="Representation layer (12 for 35M, 33 for 650M)")
    emb_group.add_argument("--toks_per_batch", type=int, default=4096,
                           help="Max tokens per batch")
    emb_group.add_argument("--truncation_seq_length", type=int, default=1022,
                           help="Max sequence length")

    # FAISS options
    idx_group = parser.add_argument_group("FAISS index options")
    idx_group.add_argument("--index_type", type=str, default="IVFPQ",
                           choices=["FlatIP", "IVFFlat", "IVFPQ", "IVFSQ"],
                           help="FAISS index type")
    idx_group.add_argument("--nlist", type=int, default=4096,
                           help="Number of IVF clusters")
    idx_group.add_argument("--pq_m", type=int, default=32,
                           help="PQ sub-vectors (IVFPQ only)")
    idx_group.add_argument("--pq_bits", type=int, default=8,
                           help="Bits per PQ code (IVFPQ only)")
    idx_group.add_argument("--nprobe", type=int, default=128,
                           help="Number of clusters to search at query time")
    idx_group.add_argument("--train_size", type=int, default=5_000_000,
                           help="Training set size for FAISS quantizers")

    # Pipeline options
    pipe_group = parser.add_argument_group("Pipeline options")
    pipe_group.add_argument("--embeddings_dir", type=str, default=None,
                            help="Dir for embedding shards (enables two-step workflow)")
    pipe_group.add_argument("--embed_only", action="store_true",
                            help="Only compute and save embeddings, don't build index")
    pipe_group.add_argument("--index_only", action="store_true",
                            help="Only build index from existing embeddings")
    pipe_group.add_argument("--shard_size", type=int, default=100_000,
                            help="Sequences per embedding shard")
    pipe_group.add_argument("--streaming", action="store_true",
                            help="One-pass mode (no multi-GPU, no resume)")
    pipe_group.add_argument("--nogpu", action="store_true",
                            help="Disable GPU")
    pipe_group.add_argument("--gpu_ids", type=str, default=None,
                            help="Comma-separated GPU IDs (e.g., '0,1,2,3')")

    # Distributed options
    dist_group = parser.add_argument_group("Distributed options (for torchrun)")
    dist_group.add_argument("--distributed", action="store_true",
                            help="Use torch.distributed (launch with torchrun)")

    return parser


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = create_parser()
    args = parser.parse_args()

    # Parse GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]

    use_gpu = not args.nogpu and torch.cuda.is_available()

    if use_gpu:
        n_gpus = torch.cuda.device_count()
        logger.info(f"CUDA available: {n_gpus} GPU(s)")
        for i in range(n_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("Running on CPU")

    # ── Distributed mode (torchrun) ──
    if args.distributed:
        if args.embeddings_dir is None:
            args.embeddings_dir = (
                str(args.index_file).replace(".index", "").replace(".faiss", "")
                + "_embeddings"
            )
        run_distributed_embedding(args)
        return

    # ── Streaming mode ──
    if args.streaming:
        logger.info("=== STREAMING MODE ===")
        embedder = ESMEmbedder(
            model_name=args.model_location,
            repr_layer=args.repr_layer,
            toks_per_batch=args.toks_per_batch,
            truncation_seq_length=args.truncation_seq_length,
            use_gpu=use_gpu,
            gpu_ids=gpu_ids,
        )
        builder = FAISSIndexBuilder(
            embed_dim=embedder.embed_dim,
            index_type=args.index_type,
            nlist=args.nlist,
            pq_m=args.pq_m,
            pq_bits=args.pq_bits,
            nprobe=args.nprobe,
            use_gpu=use_gpu,
            train_size=args.train_size,
        )
        builder.build_streaming(embedder, str(args.fasta_file), str(args.index_file))
        return

    # ── Two-step mode ──

    # Default embeddings dir
    if args.embeddings_dir is None:
        args.embeddings_dir = (
            str(args.index_file).replace(".index", "").replace(".faiss", "")
            + "_embeddings"
        )

    # Step 1: Embed
    if not args.index_only:
        logger.info(f"=== STEP 1: Embedding -> {args.embeddings_dir} ===")
        embedder = ESMEmbedder(
            model_name=args.model_location,
            repr_layer=args.repr_layer,
            toks_per_batch=args.toks_per_batch,
            truncation_seq_length=args.truncation_seq_length,
            use_gpu=use_gpu,
            gpu_ids=gpu_ids,
        )
        total_seqs, shard_paths = embedder.embed_fasta_to_shards(
            str(args.fasta_file),
            args.embeddings_dir,
            shard_size=args.shard_size,
        )

        if args.embed_only:
            logger.info("Embedding complete (--embed_only). Exiting.")
            return
    else:
        if args.embeddings_dir is None:
            logger.error("--index_only requires --embeddings_dir")
            sys.exit(1)

    # Step 2: Build index
    logger.info(f"=== STEP 2: Building FAISS index from {args.embeddings_dir} ===")

    shard_paths = sorted(glob(os.path.join(args.embeddings_dir, "shard_*.npy")))
    if not shard_paths:
        logger.error(f"No shard files found in {args.embeddings_dir}")
        sys.exit(1)
    logger.info(f"Found {len(shard_paths)} embedding shards")

    sample = np.load(shard_paths[0], mmap_mode="r")
    embed_dim = sample.shape[1]
    logger.info(f"Embedding dimension: {embed_dim}")

    builder = FAISSIndexBuilder(
        embed_dim=embed_dim,
        index_type=args.index_type,
        nlist=args.nlist,
        pq_m=args.pq_m,
        pq_bits=args.pq_bits,
        nprobe=args.nprobe,
        use_gpu=use_gpu,
        train_size=args.train_size,
    )
    builder.build_from_shards(shard_paths, str(args.index_file))

    # Consolidate labels
    label_files = sorted(glob(os.path.join(args.embeddings_dir, "shard_*.pkl")))
    if label_files:
        all_labels = []
        for lf in label_files:
            with open(lf, "rb") as f:
                all_labels.extend(pickle.load(f))
        labels_path = (
            str(args.index_file).replace(".index", "").replace(".faiss", "")
            + "_labels.pkl"
        )
        with open(labels_path, "wb") as f:
            pickle.dump(all_labels, f)
        logger.info(f"Consolidated {len(all_labels):,} labels -> {labels_path}")

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
