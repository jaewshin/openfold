#!/usr/bin/env python3 -u
"""
Build a FAISS index for protein sequences using ESM-2 mean-pooled embeddings.

Combines:
- Streaming ESM-2 embedding (memory-efficient, no full matrix in RAM)
- GPU FAISS index training and construction
- Support for FlatIP, IVF, IVFPQ, IVFSQ index types
- Handles UniRef50-scale databases (~60M sequences)

Usage:
    # FlatIP (exact search, for small databases like Pfam):
    python build_faiss_index.py esm2_t33_650M_UR50D /path/to/pfam.fasta \
        /path/to/output.index --index_type FlatIP

    # IVFPQ (compressed, for UniRef50-scale):
    python build_faiss_index.py esm2_t33_650M_UR50D /path/to/uniref50.fasta \
        /path/to/output.index --index_type IVFPQ --nlist 65536 --pq_m 32

    # IVFFlat (uncompressed IVF, good balance):
    python build_faiss_index.py esm2_t33_650M_UR50D /path/to/uniref50.fasta \
        /path/to/output.index --index_type IVFFlat --nlist 4096

    # Resume from pre-computed embeddings:
    python build_faiss_index.py esm2_t33_650M_UR50D /path/to/uniref50.fasta \
        /path/to/output.index --index_type IVFPQ --embeddings_dir /path/to/embeds/
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
# ESM-2 Embedding
# ──────────────────────────────────────────────────────────────

class ESMEmbedder:
    """
    Streams ESM-2 mean-pooled embeddings from a FASTA file.
    Handles batching, GPU transfer, and proper masking.
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        repr_layer: int = 33,
        toks_per_batch: int = 4096,
        truncation_seq_length: int = 1022,
        use_gpu: bool = True,
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
        del model  # free memory

        self.device = torch.device("cpu")
        if use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = torch.device("cuda")
            logger.info("Model transferred to GPU")

        self.embed_dim = self.model.embed_tokens.embedding_dim
        logger.info(f"Embedding dimension: {self.embed_dim}")

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

    @torch.no_grad()
    def embed_batches(self, data_loader, max_batches: Optional[int] = None):
        """
        Generator that yields (labels, mean_pooled_embeddings) for each batch.

        Args:
            data_loader: from create_dataloader
            max_batches: stop after this many batches (for training set sampling)

        Yields:
            (labels: List[str], embeddings: np.ndarray of shape (batch_size, embed_dim))
        """
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            toks = toks.to(device=self.device, non_blocking=True)
            out = self.model(toks, repr_layers=[self.repr_layer])
            reprs = out["representations"][self.repr_layer]
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
        Embed all sequences in a FASTA and save as sharded .npy files.
        Also saves sequence labels as .pkl alongside each shard.

        Returns:
            (total_sequences, list_of_shard_paths)
        """
        os.makedirs(output_dir, exist_ok=True)
        data_loader, dataset = self.create_dataloader(fasta_path)

        shard_idx = 0
        buffer_embeddings = []
        buffer_labels = []
        total_seqs = 0
        shard_paths = []

        logger.info(f"Embedding sequences -> shards in {output_dir}")

        for labels, embeddings in tqdm(
            self.embed_batches(data_loader), desc="Embedding", total=len(data_loader)
        ):
            buffer_embeddings.append(embeddings)
            buffer_labels.extend(labels)
            total_seqs += len(labels)

            # Flush shard when buffer is large enough
            current_count = sum(e.shape[0] for e in buffer_embeddings)
            if current_count >= shard_size:
                shard_path = self._save_shard(
                    output_dir, shard_idx, buffer_embeddings, buffer_labels
                )
                shard_paths.append(shard_path)
                shard_idx += 1
                buffer_embeddings = []
                buffer_labels = []

        # Final shard
        if buffer_embeddings:
            shard_path = self._save_shard(
                output_dir, shard_idx, buffer_embeddings, buffer_labels
            )
            shard_paths.append(shard_path)

        logger.info(f"Embedded {total_seqs} sequences into {len(shard_paths)} shards")

        # Save metadata
        meta = {
            "total_sequences": total_seqs,
            "embed_dim": self.embed_dim,
            "model": self.model_name,
            "repr_layer": self.repr_layer,
            "num_shards": len(shard_paths),
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return total_seqs, shard_paths

    def _save_shard(
        self,
        output_dir: str,
        shard_idx: int,
        buffer_embeddings: list,
        buffer_labels: list,
    ) -> str:
        """Save a shard of embeddings and labels to disk."""
        mat = np.concatenate(buffer_embeddings, axis=0).astype(np.float32)
        npy_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.npy")
        pkl_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.pkl")

        np.save(npy_path, mat)
        with open(pkl_path, "wb") as f:
            pickle.dump(buffer_labels, f)

        logger.info(f"  Saved shard {shard_idx}: {mat.shape[0]} vectors -> {npy_path}")
        return npy_path


# ──────────────────────────────────────────────────────────────
# FAISS Index Builder
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

        logger.info(f"Creating CPU index: faiss.index_factory({d}, '{index_key}', METRIC_INNER_PRODUCT)")
        return faiss.index_factory(d, index_key, faiss.METRIC_INNER_PRODUCT)

    def _collect_training_set(self, shard_paths: List[str]) -> np.ndarray:
        """
        Collect a training set by reservoir-sampling from shards.
        """
        logger.info(f"Collecting training set: {self.train_size} vectors from {len(shard_paths)} shards")

        # Calculate how many vectors to sample from each shard
        all_vectors = []
        remaining = self.train_size

        # First pass: count total vectors
        total_vecs = 0
        for path in shard_paths:
            mat = np.load(path, mmap_mode="r")
            total_vecs += mat.shape[0]

        # Second pass: proportionally sample from each shard
        for path in shard_paths:
            mat = np.load(path, mmap_mode="r")
            n = mat.shape[0]
            # Proportional sampling
            n_sample = max(1, int(self.train_size * n / total_vecs))
            n_sample = min(n_sample, n, remaining)

            if n_sample <= 0:
                continue

            indices = np.random.choice(n, size=n_sample, replace=False)
            # Load only the sampled rows into memory
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
            # GPU training expects torch tensors
            train_tensor = torch.from_numpy(train_set).float().contiguous()
            if torch.cuda.is_available():
                train_tensor = train_tensor.cuda()
            self.index.train(train_tensor)
        else:
            self.index = self._create_cpu_index()
            self.index.train(train_set)

        elapsed = time.time() - t0
        logger.info(f"Index trained in {elapsed:.1f}s")

    def _add_vectors(self, shard_paths: List[str]):
        """Stream vectors from shards into the trained index."""
        logger.info(f"Adding vectors from {len(shard_paths)} shards...")
        total_added = 0

        for shard_idx, path in enumerate(tqdm(shard_paths, desc="Adding shards")):
            mat = np.load(path).astype(np.float32)

            if self.use_gpu:
                # Add in sub-chunks to avoid GPU OOM
                chunk_size = 100_000
                for start in range(0, mat.shape[0], chunk_size):
                    end = min(start + chunk_size, mat.shape[0])
                    chunk = torch.from_numpy(mat[start:end]).float().contiguous()
                    if torch.cuda.is_available():
                        chunk = chunk.cuda()
                    self.index.add(chunk)
            else:
                self.index.add(mat)

            total_added += mat.shape[0]

            if (shard_idx + 1) % 10 == 0:
                logger.info(f"  Added {total_added:,} vectors so far")

        logger.info(f"Total vectors in index: {self.index.ntotal:,}")

    def build_from_shards(
        self,
        shard_paths: List[str],
        save_path: str,
    ):
        """
        Full pipeline: collect training set, train, add all vectors, save.

        Args:
            shard_paths: list of .npy files containing embeddings
            save_path: output path for the FAISS index
        """
        # 1. Collect training set
        if self.index_type != "FLATIP":
            train_set = self._collect_training_set(shard_paths)
            # 2. Train
            self._train_index(train_set)
            del train_set  # free memory
        else:
            # FlatIP doesn't need training
            if self.use_gpu:
                self.index = self._create_gpu_index()
            else:
                self.index = self._create_cpu_index()

        # 3. Set nprobe for IVF indices
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

        # 4. Add all vectors
        self._add_vectors(shard_paths)

        # 5. Save (move to CPU first if on GPU)
        self.save_index(save_path)

    def build_streaming(
        self,
        embedder: ESMEmbedder,
        fasta_path: str,
        save_path: str,
    ):
        """
        One-pass pipeline: embed + build index without saving shards to disk.
        More memory-efficient for very large databases but can't resume.

        Steps:
            1. First pass: sample training vectors
            2. Train index
            3. Second pass: stream all vectors into the index
        """
        data_loader, dataset = embedder.create_dataloader(fasta_path)

        # ── Phase 1: Collect training set ──
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

        # ── Phase 2: Stream all vectors into index ──
        # Re-create dataloader for full pass
        data_loader, _ = embedder.create_dataloader(fasta_path)
        logger.info("Phase 2: Streaming all sequences into the index...")

        all_labels = []
        total_added = 0

        for labels, embeddings in tqdm(
            embedder.embed_batches(data_loader), desc="Indexing", total=len(data_loader)
        ):
            if self.use_gpu:
                vec = torch.from_numpy(embeddings).float().contiguous().cuda()
                self.index.add(vec)
            else:
                self.index.add(embeddings.astype(np.float32))

            all_labels.extend(labels)
            total_added += embeddings.shape[0]

        logger.info(f"Total vectors in index: {self.index.ntotal:,}")

        # Save index
        self.save_index(save_path)

        # Save label mapping (sequence_id -> index position)
        labels_path = save_path.replace(".index", "").replace(".faiss", "") + "_labels.pkl"
        with open(labels_path, "wb") as f:
            pickle.dump(all_labels, f)
        logger.info(f"Labels saved to {labels_path}")

    def save_index(self, path: str):
        """Save the FAISS index to disk (CPU format)."""
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
    def load_index(
        path: str,
        use_gpu: bool = False,
        nprobe: int = 128,
    ) -> faiss.Index:
        """Load a FAISS index from disk, optionally moving to GPU."""
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
  # Small database (Pfam), exact search:
  python build_faiss_index.py esm2_t33_650M_UR50D pfam.fasta pfam.index \\
      --index_type FlatIP

  # Large database (UniRef50), compressed:
  python build_faiss_index.py esm2_t33_650M_UR50D uniref50.fasta uniref50.index \\
      --index_type IVFPQ --nlist 65536 --pq_m 32 --train_size 500000

  # Two-step: embed first, then build index:
  python build_faiss_index.py esm2_t33_650M_UR50D uniref50.fasta uniref50.index \\
      --index_type IVFPQ --embeddings_dir ./embeds/ --embed_only
  python build_faiss_index.py esm2_t33_650M_UR50D uniref50.fasta uniref50.index \\
      --index_type IVFPQ --embeddings_dir ./embeds/ --index_only
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
    emb_group.add_argument("--repr_layer", type=int, default=33, help="Representation layer (33 for 650M, 48 for 3B)")
    emb_group.add_argument("--toks_per_batch", type=int, default=4096, help="Max tokens per batch")
    emb_group.add_argument("--truncation_seq_length", type=int, default=1022, help="Max sequence length")

    # FAISS options
    idx_group = parser.add_argument_group("FAISS index options")
    idx_group.add_argument("--index_type", type=str, default="IVFPQ",
                           choices=["FlatIP", "IVFFlat", "IVFPQ", "IVFSQ"],
                           help="FAISS index type")
    idx_group.add_argument("--nlist", type=int, default=4096, help="Number of IVF clusters")
    idx_group.add_argument("--pq_m", type=int, default=32, help="PQ sub-vectors (IVFPQ only)")
    idx_group.add_argument("--pq_bits", type=int, default=8, help="Bits per PQ code (IVFPQ only)")
    idx_group.add_argument("--nprobe", type=int, default=128, help="Number of clusters to search")
    idx_group.add_argument("--train_size", type=int, default=256000, help="Training set size")

    # Pipeline options
    pipe_group = parser.add_argument_group("Pipeline options")
    pipe_group.add_argument("--embeddings_dir", type=str, default=None,
                            help="Dir for embedding shards (enables two-step workflow)")
    pipe_group.add_argument("--embed_only", action="store_true",
                            help="Only compute and save embeddings, don't build index")
    pipe_group.add_argument("--index_only", action="store_true",
                            help="Only build index from existing embeddings in --embeddings_dir")
    pipe_group.add_argument("--shard_size", type=int, default=100_000,
                            help="Sequences per embedding shard")
    pipe_group.add_argument("--streaming", action="store_true",
                            help="One-pass mode: embed and index without saving shards")
    pipe_group.add_argument("--nogpu", action="store_true", help="Disable GPU")

    return parser


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = create_parser()
    args = parser.parse_args()
    use_gpu = not args.nogpu and torch.cuda.is_available()

    if args.nogpu:
        logger.info("GPU disabled by --nogpu flag")
    elif not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")

    # ── Streaming mode: one-pass embed + index ──
    if args.streaming:
        logger.info("=== STREAMING MODE: embed + index in one pass ===")
        embedder = ESMEmbedder(
            model_name=args.model_location,
            repr_layer=args.repr_layer,
            toks_per_batch=args.toks_per_batch,
            truncation_seq_length=args.truncation_seq_length,
            use_gpu=use_gpu,
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

    # Step 1: Embed (if not index_only)
    if not args.index_only:
        if args.embeddings_dir is None and not args.streaming:
            # Default: use temp embeddings dir next to index file
            args.embeddings_dir = str(args.index_file).replace(".index", "").replace(".faiss", "") + "_embeddings"

        logger.info(f"=== STEP 1: Embedding sequences -> {args.embeddings_dir} ===")
        embedder = ESMEmbedder(
            model_name=args.model_location,
            repr_layer=args.repr_layer,
            toks_per_batch=args.toks_per_batch,
            truncation_seq_length=args.truncation_seq_length,
            use_gpu=use_gpu,
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

    # Step 2: Build index from shards
    logger.info(f"=== STEP 2: Building FAISS index from {args.embeddings_dir} ===")

    shard_paths = sorted(glob(os.path.join(args.embeddings_dir, "shard_*.npy")))
    if not shard_paths:
        logger.error(f"No shard files found in {args.embeddings_dir}")
        sys.exit(1)
    logger.info(f"Found {len(shard_paths)} embedding shards")

    # Infer embed dim from first shard
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

    # Also consolidate labels
    label_files = sorted(glob(os.path.join(args.embeddings_dir, "shard_*.pkl")))
    if label_files:
        all_labels = []
        for lf in label_files:
            with open(lf, "rb") as f:
                all_labels.extend(pickle.load(f))
        labels_path = str(args.index_file).replace(".index", "").replace(".faiss", "") + "_labels.pkl"
        with open(labels_path, "wb") as f:
            pickle.dump(all_labels, f)
        logger.info(f"Consolidated {len(all_labels):,} labels -> {labels_path}")

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()