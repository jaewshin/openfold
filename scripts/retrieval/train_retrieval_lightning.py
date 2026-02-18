#!/usr/bin/env python3
"""Memory-aware Lightning pipeline for retrieval-augmented structure training.

Key design goals:
- Use `download_structures.py` fixture format directly.
- Keep FAISS indexes on CPU and load lazily.
- Avoid loading full retrieval embedding tensors into GPU memory.
- Support structure-only retrieval when sequence index is unavailable.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import math
import os
import re
import shutil
import struct
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

# Allow running directly from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.model.retrieval_fusion import CrossAttentionFusion
from openfold.utils.loss import AlphaFoldLoss
from openfold.utils.tensor_utils import tensor_tree_map

from scripts.retrieval.retrieval_data import RetrievalDataModule, build_manifest, write_manifest_jsonl

logger = logging.getLogger(__name__)


class LazyFaissRetriever:
    """Lazy CPU FAISS loader with per-query reconstruct-based retrieval."""

    def __init__(
        self,
        index_path: str,
        top_k: int = 8,
        nprobe: int = 64,
    ):
        self.index_path = index_path
        self.top_k = top_k
        self.nprobe = nprobe
        self._index = None
        self.dim: Optional[int] = None

    def _ensure_loaded(self):
        if self._index is not None:
            return
        try:
            import faiss
        except Exception as exc:
            raise RuntimeError(
                "faiss is required for retrieval. Install faiss-cpu/faiss-gpu in the active environment."
            ) from exc

        logger.info("Loading FAISS index from %s", self.index_path)
        self._index = faiss.read_index(self.index_path)
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = self.nprobe
        self.dim = int(self._index.d)
        logger.info("Loaded index (dim=%d, ntotal=%d)", self.dim, int(self._index.ntotal))

    def search(self, query_vec: np.ndarray, top_k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Search one query vector and reconstruct top-k vectors from the index."""
        self._ensure_loaded()
        assert self._index is not None
        assert self.dim is not None

        k = int(top_k or self.top_k)
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError(f"Query dim mismatch: got {q.shape[1]} expected {self.dim}")

        distances, indices = self._index.search(q, k)
        d = distances[0].astype(np.float32, copy=False)
        i = indices[0].astype(np.int64, copy=False)

        vectors = np.zeros((k, self.dim), dtype=np.float32)
        for j, idx in enumerate(i.tolist()):
            if idx < 0:
                continue
            try:
                vectors[j] = self._index.reconstruct(int(idx))
            except Exception:
                # Keep zero vector when reconstruct is unavailable for an entry.
                pass
        return d, i, vectors


def _sanitize_aa_sequence(seq: str) -> str:
    seq = str(seq).upper()
    # Replace unexpected symbols so tokenizers do not crash on malformed records.
    return re.sub(r"[^A-Z]", "X", seq)


class ESM2QueryEncoder(nn.Module):
    """Trainable ESM2 query encoder matching the ESM2-35M FAISS embedding space."""

    def __init__(
        self,
        model_name: str = "esm2_t12_35M_UR50D",
        repr_layer: int = 12,
        truncation_seq_length: int = 1022,
        normalize: bool = True,
    ):
        super().__init__()
        try:
            from esm import pretrained
        except Exception as exc:
            raise RuntimeError(
                "Failed to import `esm`. Install fair-esm in the active environment."
            ) from exc

        model, alphabet = pretrained.load_model_and_alphabet(model_name)
        self.model = model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter(
            truncation_seq_length=truncation_seq_length
        )
        self.repr_layer = int(repr_layer)
        self.normalize = normalize
        self.truncation_seq_length = truncation_seq_length
        self.output_dim = int(getattr(model, "embed_dim", 480))

    def forward(self, sequences: Sequence[str]) -> torch.Tensor:
        if not sequences:
            return torch.empty(0, self.output_dim, device=next(self.parameters()).device)

        batch = [(str(i), _sanitize_aa_sequence(s)) for i, s in enumerate(sequences)]
        _, _, toks = self.batch_converter(batch)
        device = next(self.model.parameters()).device
        toks = toks.to(device=device, non_blocking=True)

        out = self.model(toks, repr_layers=[self.repr_layer], return_contacts=False)
        reps = out["representations"][self.repr_layer]  # [B, T, D]
        mask = (
            (toks != self.alphabet.padding_idx)
            & (toks != self.alphabet.cls_idx)
            & (toks != self.alphabet.eos_idx)
        ).float()
        pooled = (reps * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled


class TMVec2QueryEncoder(nn.Module):
    """Trainable TMVec-2s query encoder matching the TMVec-2s FAISS embedding space."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        max_length: int = 1022,
        normalize: bool = True,
    ):
        super().__init__()
        tmvec_root = REPO_ROOT / "tmvec-bench"
        if str(tmvec_root) not in sys.path:
            sys.path.insert(0, str(tmvec_root))
        try:
            from src.model.tmvec2_student_model import StudentModel, encode_sequence
        except Exception as exc:
            raise RuntimeError(
                "Failed to import TMVec-2s model from tmvec-bench. "
                f"Expected path: {tmvec_root}"
            ) from exc

        self._encode_sequence = encode_sequence
        self.model = StudentModel()
        self.max_length = int(max_length)
        self.normalize = normalize
        self.output_dim = 512

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            else:
                state_dict = ckpt
            self.model.load_state_dict(state_dict, strict=True)
            logger.info("Loaded TMVec-2s checkpoint from %s", checkpoint_path)
        else:
            logger.warning(
                "TMVec-2s query encoder initialized without checkpoint. "
                "Provide --retriever_tmvec_checkpoint to match index space."
            )

    def forward(self, sequences: Sequence[str]) -> torch.Tensor:
        if not sequences:
            return torch.empty(0, self.output_dim, device=next(self.parameters()).device)

        toks = torch.stack(
            [self._encode_sequence(_sanitize_aa_sequence(s), self.max_length) for s in sequences],
            dim=0,
        )
        device = next(self.model.parameters()).device
        toks = toks.to(device=device, non_blocking=True)
        pooled = self.model.seq_encoder(toks)
        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled


class RowIdLookup:
    """Random-access row-id lookup from a line-based ids file."""

    def __init__(self, ids_path: str, cache_size: int = 8192):
        self.original_path = Path(ids_path)
        self.cache_size = int(cache_size)
        self._cache: "OrderedDict[int, str]" = OrderedDict()

        if self.original_path.suffix == ".pkl":
            txt_candidate = self.original_path.with_suffix(".txt")
            if not txt_candidate.exists():
                raise ValueError(
                    f"Unsupported ids mapping format for random access: {self.original_path}. "
                    "Provide a line-based .txt/.txt.gz map (one sequence id per line, row-aligned with FAISS)."
                )
            self.text_path = txt_candidate
        elif self.original_path.suffix == ".gz":
            text_path = self.original_path.with_suffix("")
            if not text_path.exists():
                logger.info("Decompressing ids map %s -> %s", self.original_path, text_path)
                with gzip.open(self.original_path, "rb") as src, open(text_path, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            self.text_path = text_path
        else:
            self.text_path = self.original_path

        if not self.text_path.exists():
            raise FileNotFoundError(f"IDs text path not found: {self.text_path}")

        self.offsets_path = self.text_path.with_suffix(self.text_path.suffix + ".offsets.u64")
        if not self.offsets_path.exists():
            self._build_offsets()

        self._offsets = np.memmap(self.offsets_path, mode="r", dtype=np.uint64)
        self._fh = open(self.text_path, "rb", buffering=1024 * 1024)

    def _build_offsets(self):
        logger.info("Building ids offset table: %s", self.offsets_path)
        with open(self.text_path, "rb", buffering=1024 * 1024) as f, open(self.offsets_path, "wb") as out:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                out.write(struct.pack("<Q", pos))

    def __len__(self) -> int:
        return int(self._offsets.shape[0])

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass

    def _cache_put(self, idx: int, value: str):
        self._cache[idx] = value
        self._cache.move_to_end(idx)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def get(self, row_idx: int) -> Optional[str]:
        row_idx = int(row_idx)
        if row_idx < 0 or row_idx >= len(self):
            return None
        cached = self._cache.get(row_idx)
        if cached is not None:
            self._cache.move_to_end(row_idx)
            return cached
        offset = int(self._offsets[row_idx])
        self._fh.seek(offset)
        line = self._fh.readline().decode("utf-8", errors="replace").strip()
        if not line:
            return None
        # Normalize to FASTA-style identifier token.
        value = line.split()[0]
        self._cache_put(row_idx, value)
        return value


class FastaSequenceStore:
    """ID->sequence lookup backed by Biopython SeqIO sqlite index."""

    def __init__(self, fasta_path: str, index_db_path: str, cache_size: int = 8192):
        try:
            from Bio import SeqIO
        except Exception as exc:
            raise RuntimeError("Biopython is required for raw-sequence retrieval pipeline.") from exc

        self.fasta_path = str(fasta_path)
        self.index_db_path = str(index_db_path)
        self.cache_size = int(cache_size)
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        logger.info(
            "Opening FASTA index db %s (fasta=%s). "
            "If this is the first run, index creation can take a long time.",
            self.index_db_path,
            self.fasta_path,
        )
        self._records = SeqIO.index_db(self.index_db_path, [self.fasta_path], "fasta")

    def close(self):
        try:
            self._records.close()
        except Exception:
            pass

    def _cache_put(self, key: str, value: str):
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def get(self, seq_id: str) -> Optional[str]:
        cached = self._cache.get(seq_id)
        if cached is not None:
            self._cache.move_to_end(seq_id)
            return cached
        rec = self._records.get(seq_id)
        if rec is None:
            return None
        seq = str(rec.seq)
        self._cache_put(seq_id, seq)
        return seq


class FrozenESM1bEmbedder:
    """Frozen ESM-1b encoder used to embed retrieved raw sequences for pipeline B."""

    def __init__(
        self,
        model_name: str = "esm1b_t33_650M_UR50S",
        repr_layer: int = 33,
        truncation_seq_length: int = 1022,
        device: str = "cpu",
        cache_size: int = 4096,
    ):
        try:
            from esm import pretrained
        except Exception as exc:
            raise RuntimeError(
                "Failed to import `esm`. Install fair-esm in the active environment."
            ) from exc

        self.device = torch.device(device if device != "cuda" else ("cuda" if torch.cuda.is_available() else "cpu"))
        model, alphabet = pretrained.load_model_and_alphabet(model_name)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.to(self.device)
        self.repr_layer = int(repr_layer)
        self.truncation_seq_length = int(truncation_seq_length)
        self.batch_converter = alphabet.get_batch_converter(
            truncation_seq_length=self.truncation_seq_length
        )
        self.embed_dim = int(getattr(model, "embed_dim", 1280))
        self.cache_size = int(cache_size)
        self._cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        logger.info("Loaded frozen ESM1b retriever embedder on %s", self.device)

    def _cache_put(self, seq_id: str, emb: torch.Tensor):
        self._cache[seq_id] = emb
        self._cache.move_to_end(seq_id)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    @torch.no_grad()
    def embed(self, seq_pairs: Sequence[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        pending: List[Tuple[str, str]] = []
        for seq_id, seq in seq_pairs:
            cached = self._cache.get(seq_id)
            if cached is not None:
                self._cache.move_to_end(seq_id)
                out[seq_id] = cached
            else:
                pending.append((seq_id, _sanitize_aa_sequence(seq)))

        if pending:
            batch = [(seq_id, seq) for seq_id, seq in pending]
            _, _, toks = self.batch_converter(batch)
            toks = toks.to(self.device, non_blocking=True)
            reps = self.model(toks, repr_layers=[self.repr_layer], return_contacts=False)["representations"][
                self.repr_layer
            ]
            reps = reps.float().cpu()

            for i, (seq_id, seq) in enumerate(pending):
                seq_len = min(len(seq), self.truncation_seq_length)
                emb = reps[i, 1 : 1 + seq_len].contiguous()
                self._cache_put(seq_id, emb)
                out[seq_id] = emb

        return out


class RetrievalAugmentedLightningModule(pl.LightningModule):
    """Lightning module with frozen OpenFold + FAISS retrieval fusion."""

    def __init__(
        self,
        config_preset: str = "seqemb_initial_training",
        seq_embedding_dim: int = 1280,
        top_k: int = 8,
        lr: float = 1e-4,
        struct_index_path: Optional[str] = None,
        struct_index_dim: int = 512,
        seq_index_path: Optional[str] = None,
        seq_index_dim: int = 1280,
        nprobe: int = 64,
        fusion_heads: int = 8,
        fusion_dropout: float = 0.0,
        retrieval_ablation: str = "both",
        retrieval_pipeline: str = "legacy",
        retriever_esm2_model_name: str = "esm2_t12_35M_UR50D",
        retriever_esm2_repr_layer: int = 12,
        retriever_esm2_max_len: int = 1022,
        retriever_tmvec_checkpoint: Optional[str] = None,
        retriever_tmvec_max_len: int = 1022,
        retriever_normalize_queries: bool = True,
        seq_index_ids_path: Optional[str] = None,
        struct_index_ids_path: Optional[str] = None,
        seq_db_fasta_path: Optional[str] = None,
        struct_db_fasta_path: Optional[str] = None,
        seq_db_fasta_index_db: Optional[str] = None,
        struct_db_fasta_index_db: Optional[str] = None,
        retrieved_esm1b_model_name: str = "esm1b_t33_650M_UR50S",
        retrieved_esm1b_repr_layer: int = 33,
        retrieved_esm1b_max_len: int = 1022,
        retrieved_esm1b_device: str = "cpu",
        retrieved_esm1b_cache_size: int = 4096,
        openfold_checkpoint: Optional[str] = None,
        freeze_backbone: bool = True,
        train_openfold_all: bool = False,
        train_input_embedder: bool = False,
        train_recycling_embedder: bool = False,
        train_template_embedder: bool = False,
        train_extra_msa_embedder: bool = False,
        train_extra_msa_stack: bool = False,
        train_evoformer: bool = False,
        train_structure_module: bool = False,
        train_aux_heads: bool = False,
        low_prec: bool = False,
        backbone_factory: Optional[Callable] = None,
        loss_factory: Optional[Callable] = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["backbone_factory", "loss_factory"]
        )
        self.seq_embedding_dim = seq_embedding_dim
        self.top_k = top_k
        self.lr = lr
        self.retrieval_ablation = retrieval_ablation
        self.retrieval_pipeline = retrieval_pipeline
        self.freeze_backbone = freeze_backbone
        self.train_openfold_all = train_openfold_all
        requested_modules: Set[str] = set()
        if train_openfold_all:
            requested_modules.update({
                "input_embedder",
                "recycling_embedder",
                "template_embedder",
                "extra_msa_embedder",
                "extra_msa_stack",
                "evoformer",
                "structure_module",
                "aux_heads",
            })
        if train_input_embedder:
            requested_modules.add("input_embedder")
        if train_recycling_embedder:
            requested_modules.add("recycling_embedder")
        if train_template_embedder:
            requested_modules.add("template_embedder")
        if train_extra_msa_embedder:
            requested_modules.add("extra_msa_embedder")
        if train_extra_msa_stack:
            requested_modules.add("extra_msa_stack")
        if train_evoformer:
            requested_modules.add("evoformer")
        if train_structure_module:
            requested_modules.add("structure_module")
        if train_aux_heads:
            requested_modules.add("aux_heads")
        self._openfold_trainable_modules = sorted(requested_modules)

        if retrieval_ablation not in {"both", "seq_only", "struct_only"}:
            raise ValueError(
                f"Unsupported retrieval_ablation={retrieval_ablation!r}. "
                "Expected one of {'both', 'seq_only', 'struct_only'}."
            )
        if retrieval_pipeline not in {"legacy", "embed_project", "rawseq_esm1b"}:
            raise ValueError(
                f"Unsupported retrieval_pipeline={retrieval_pipeline!r}. "
                "Expected one of {'legacy', 'embed_project', 'rawseq_esm1b'}."
            )

        self.config = model_config(config_preset, train=True, low_prec=low_prec)
        # Template-free for this retrieval fixture (no alignment/template dependency).
        self.config.model.template.enabled = False
        self.config.data.common.use_templates = False

        if backbone_factory is None:
            self.openfold = AlphaFold(self.config)
        else:
            self.openfold = backbone_factory(self.config)

        if openfold_checkpoint:
            sd = torch.load(openfold_checkpoint, map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
            # Accept either plain AlphaFold or wrapped checkpoints.
            remapped = {}
            for k, v in sd.items():
                if k.startswith("model.openfold."):
                    remapped[k[len("model.openfold."):]] = v
                elif k.startswith("model."):
                    remapped[k[len("model."):]] = v
                else:
                    remapped[k] = v
            missing, unexpected = self.openfold.load_state_dict(remapped, strict=False)
            logger.info(
                "Loaded OpenFold checkpoint=%s (missing=%d unexpected=%d)",
                openfold_checkpoint, len(missing), len(unexpected)
            )

        if freeze_backbone:
            for p in self.openfold.parameters():
                p.requires_grad_(False)
            self.openfold.eval()

            if self.train_openfold_all:
                for p in self.openfold.parameters():
                    p.requires_grad_(True)
                self.openfold.train()
                self._openfold_trainable_modules = []
                logger.info("OpenFold backbone fully trainable (train_openfold_all=True).")
            else:
                enabled_modules = []
                missing_modules = []
                for module_name in self._openfold_trainable_modules:
                    module = getattr(self.openfold, module_name, None)
                    if module is None:
                        missing_modules.append(module_name)
                        continue
                    for p in module.parameters():
                        p.requires_grad_(True)
                    module.train()
                    enabled_modules.append(module_name)

                self._openfold_trainable_modules = enabled_modules
                if missing_modules:
                    logger.warning(
                        "Requested OpenFold trainable modules not found and ignored: %s",
                        ", ".join(missing_modules),
                    )
                if enabled_modules:
                    logger.info(
                        "OpenFold backbone frozen except modules: %s",
                        ", ".join(enabled_modules),
                    )
        elif self._openfold_trainable_modules:
            logger.info(
                "freeze_backbone=False: full OpenFold backbone is trainable; "
                "module-specific train flags are redundant."
            )

        self.loss_fn = AlphaFoldLoss(self.config.loss) if loss_factory is None else loss_factory(self.config.loss)

        self.seq_retriever = LazyFaissRetriever(seq_index_path, top_k=top_k, nprobe=nprobe) if seq_index_path else None
        self.struct_retriever = (
            LazyFaissRetriever(struct_index_path, top_k=top_k, nprobe=nprobe) if struct_index_path else None
        )

        self.seq_query_proj = nn.Linear(seq_embedding_dim, seq_index_dim)
        self.struct_query_proj = nn.Linear(seq_embedding_dim, struct_index_dim)
        self.seq_db_proj = nn.Linear(seq_index_dim, seq_embedding_dim)
        self.struct_db_proj = nn.Linear(struct_index_dim, seq_embedding_dim)

        self.seq_fusion = CrossAttentionFusion(
            emb_dim=seq_embedding_dim,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
        )
        self.struct_fusion = CrossAttentionFusion(
            emb_dim=seq_embedding_dim,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
        )
        self.source_mix_logits = nn.Parameter(torch.zeros(2))

        # Backward-compatible legacy query projectors.
        # New pipelines use explicit trainable query encoders below.
        self.seq_query_encoder: Optional[nn.Module] = None
        self.struct_query_encoder: Optional[nn.Module] = None

        if self.retrieval_pipeline in {"embed_project", "rawseq_esm1b"}:
            if self.seq_retriever is not None:
                self.seq_query_encoder = ESM2QueryEncoder(
                    model_name=retriever_esm2_model_name,
                    repr_layer=retriever_esm2_repr_layer,
                    truncation_seq_length=retriever_esm2_max_len,
                    normalize=retriever_normalize_queries,
                )
            if self.struct_retriever is not None:
                self.struct_query_encoder = TMVec2QueryEncoder(
                    checkpoint_path=retriever_tmvec_checkpoint,
                    max_length=retriever_tmvec_max_len,
                    normalize=retriever_normalize_queries,
                )

        self.seq_id_lookup: Optional[RowIdLookup] = None
        self.struct_id_lookup: Optional[RowIdLookup] = None
        self.seq_sequence_store: Optional[FastaSequenceStore] = None
        self.struct_sequence_store: Optional[FastaSequenceStore] = None
        self.retrieved_esm1b_embedder: Optional[FrozenESM1bEmbedder] = None

        if self.retrieval_pipeline == "rawseq_esm1b":
            if self.seq_retriever is not None:
                if not seq_index_ids_path:
                    raise ValueError(
                        "Pipeline rawseq_esm1b with sequence retrieval requires --seq_index_ids_path."
                    )
                self.seq_id_lookup = RowIdLookup(seq_index_ids_path)
            if self.struct_retriever is not None:
                if not struct_index_ids_path:
                    raise ValueError(
                        "Pipeline rawseq_esm1b with structure retrieval requires --struct_index_ids_path."
                    )
                self.struct_id_lookup = RowIdLookup(struct_index_ids_path)

            # Use one or two FASTA stores depending on whether sources share DB paths.
            store_cache: Dict[Tuple[str, str], FastaSequenceStore] = {}

            def _get_or_make_store(fasta_path: Optional[str], idx_db: Optional[str], source_name: str):
                if fasta_path is None or idx_db is None:
                    raise ValueError(
                        f"Pipeline rawseq_esm1b requires --{source_name}_db_fasta_path and "
                        f"--{source_name}_db_fasta_index_db."
                    )
                key = (fasta_path, idx_db)
                if key not in store_cache:
                    store_cache[key] = FastaSequenceStore(
                        fasta_path=fasta_path,
                        index_db_path=idx_db,
                    )
                return store_cache[key]

            if self.seq_retriever is not None:
                seq_fasta = seq_db_fasta_path or struct_db_fasta_path
                seq_idx_db = seq_db_fasta_index_db or struct_db_fasta_index_db
                self.seq_sequence_store = _get_or_make_store(
                    fasta_path=seq_fasta,
                    idx_db=seq_idx_db,
                    source_name="seq",
                )
            if self.struct_retriever is not None:
                struct_fasta = struct_db_fasta_path or seq_db_fasta_path
                struct_idx_db = struct_db_fasta_index_db or seq_db_fasta_index_db
                self.struct_sequence_store = _get_or_make_store(
                    fasta_path=struct_fasta,
                    idx_db=struct_idx_db,
                    source_name="struct",
                )

            if self.seq_retriever is not None or self.struct_retriever is not None:
                self.retrieved_esm1b_embedder = FrozenESM1bEmbedder(
                    model_name=retrieved_esm1b_model_name,
                    repr_layer=retrieved_esm1b_repr_layer,
                    truncation_seq_length=retrieved_esm1b_max_len,
                    device=retrieved_esm1b_device,
                    cache_size=retrieved_esm1b_cache_size,
                )

    def on_train_start(self):
        # Keep frozen OpenFold blocks in eval mode while selected submodules remain trainable.
        if self.freeze_backbone:
            if self.train_openfold_all:
                self.openfold.train()
                return
            self.openfold.eval()
            for module_name in self._openfold_trainable_modules:
                module = getattr(self.openfold, module_name, None)
                if module is not None:
                    module.train()

    def _enabled_sources(self) -> Tuple[bool, bool]:
        use_seq = self.retrieval_ablation in {"both", "seq_only"}
        use_struct = self.retrieval_ablation in {"both", "struct_only"}
        if use_seq and self.seq_retriever is None:
            use_seq = False
        if use_struct and self.struct_retriever is None:
            use_struct = False
        if not use_seq and not use_struct:
            raise ValueError(
                "No active retriever source. Provide struct_index_path and/or seq_index_path "
                "or change --retrieval_ablation."
            )
        return use_seq, use_struct

    @staticmethod
    def _pool_query(tokens: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return tokens.mean(dim=0)
        m = mask.float().unsqueeze(-1)
        return (tokens * m).sum(dim=0) / m.sum(dim=0).clamp(min=1.0)

    def teardown(self, stage: Optional[str] = None) -> None:
        for lookup in (self.seq_id_lookup, self.struct_id_lookup):
            if lookup is not None:
                lookup.close()
        for store in (self.seq_sequence_store, self.struct_sequence_store):
            if store is not None:
                store.close()
        super().teardown(stage)

    @staticmethod
    def _search_index(
        query_vec: torch.Tensor,
        retriever: LazyFaissRetriever,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Search FAISS and build differentiable scores over fixed top-k candidates."""
        q_np = query_vec.detach().float().cpu().numpy()
        _, indices_np, vectors_np = retriever.search(q_np, top_k=top_k)

        device = query_vec.device
        indices = torch.from_numpy(indices_np).to(device=device, dtype=torch.long)
        valid = indices >= 0
        retrieved_keys = torch.from_numpy(vectors_np).to(device=device, dtype=torch.float32)

        logits = torch.matmul(retrieved_keys, query_vec) / math.sqrt(max(1, query_vec.shape[-1]))
        if valid.any():
            logits = logits.masked_fill(~valid, -1e9)
            scores = torch.softmax(logits, dim=0)
        else:
            scores = torch.zeros_like(logits)
        return scores, indices, retrieved_keys, valid

    @staticmethod
    def _prepare_embed_project_retrieved(
        retrieved_keys: torch.Tensor,
        valid: torch.Tensor,
        db_proj: nn.Linear,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # One vector per candidate -> shape [K, 1, D] for cross-attention.
        retrieved = db_proj(retrieved_keys.unsqueeze(1))
        retrieved_masks = valid.float().unsqueeze(-1)
        return retrieved, retrieved_masks

    def _lookup_source_components(self, source: str) -> Tuple[RowIdLookup, FastaSequenceStore]:
        if source == "seq":
            id_lookup = self.seq_id_lookup
            store = self.seq_sequence_store
        elif source == "struct":
            id_lookup = self.struct_id_lookup
            store = self.struct_sequence_store
        else:
            raise ValueError(f"Unsupported retrieval source: {source}")
        if id_lookup is None or store is None:
            raise RuntimeError(
                f"Pipeline rawseq_esm1b missing id/sequence store for source={source}."
            )
        return id_lookup, store

    def _prepare_rawseq_retrieved(
        self,
        source: str,
        indices: torch.Tensor,
        scores: torch.Tensor,
        valid: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert FAISS row ids -> raw sequences -> frozen ESM1b embeddings."""
        if self.retrieved_esm1b_embedder is None:
            raise RuntimeError("rawseq_esm1b mode requires initialized retrieved_esm1b_embedder")
        id_lookup, seq_store = self._lookup_source_components(source)

        candidate_ids: List[Optional[str]] = []
        candidate_seqs: List[Optional[str]] = []
        content_valid = valid.clone()
        for j, idx in enumerate(indices.tolist()):
            if not bool(valid[j].item()):
                candidate_ids.append(None)
                candidate_seqs.append(None)
                continue
            seq_id = id_lookup.get(int(idx))
            if seq_id is None:
                content_valid[j] = False
                candidate_ids.append(None)
                candidate_seqs.append(None)
                continue
            seq = seq_store.get(seq_id)
            if not seq:
                content_valid[j] = False
                candidate_ids.append(None)
                candidate_seqs.append(None)
                continue
            candidate_ids.append(seq_id)
            candidate_seqs.append(seq)

        pairs = [
            (sid, seq)
            for sid, seq, ok in zip(candidate_ids, candidate_seqs, content_valid.tolist())
            if ok and sid is not None and seq is not None
        ]
        emb_map = self.retrieved_esm1b_embedder.embed(pairs) if pairs else {}

        max_len = 1
        for sid, ok in zip(candidate_ids, content_valid.tolist()):
            if not ok or sid is None:
                continue
            emb = emb_map.get(sid)
            if emb is not None:
                max_len = max(max_len, int(emb.shape[0]))

        retrieved = torch.zeros(
            indices.shape[0], max_len, self.seq_embedding_dim, device=device, dtype=torch.float32
        )
        retrieved_masks = torch.zeros(indices.shape[0], max_len, device=device, dtype=torch.float32)

        for j, (sid, ok) in enumerate(zip(candidate_ids, content_valid.tolist())):
            if not ok or sid is None:
                continue
            emb = emb_map.get(sid)
            if emb is None:
                content_valid[j] = False
                continue
            n = min(int(emb.shape[0]), max_len)
            retrieved[j, :n] = emb[:n].to(device=device, dtype=torch.float32)
            retrieved_masks[j, :n] = 1.0

        if content_valid.any():
            scores = scores.masked_fill(~content_valid, -1e9)
            scores = torch.softmax(scores, dim=0)
        else:
            scores = torch.zeros_like(scores)

        return retrieved, retrieved_masks, scores

    def _run_source_legacy(
        self,
        query_tokens: torch.Tensor,
        query_mask: Optional[torch.Tensor],
        query_proj: nn.Linear,
        db_proj: nn.Linear,
        fusion: CrossAttentionFusion,
        retriever: LazyFaissRetriever,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = self._pool_query(query_tokens, query_mask)
        q_vec = query_proj(pooled).detach().cpu().numpy()
        scores_np, indices_np, vectors_np = retriever.search(q_vec, top_k=self.top_k)

        device = query_tokens.device
        valid = torch.from_numpy((indices_np >= 0)).to(device=device)
        scores = torch.from_numpy(scores_np).to(device=device, dtype=torch.float32)
        if valid.any():
            scores = scores.masked_fill(~valid, -1e9)
            scores = torch.softmax(scores, dim=0)
        else:
            # If no valid hits are returned, the source contributes nothing.
            scores = torch.zeros_like(scores)

        retrieved = torch.from_numpy(vectors_np).to(device=device, dtype=torch.float32).unsqueeze(1)
        retrieved = db_proj(retrieved)
        retrieved_masks = valid.float().unsqueeze(-1)

        fused = fusion(query_tokens, retrieved, scores, retrieved_masks=retrieved_masks)
        indices = torch.from_numpy(indices_np).to(device=device, dtype=torch.long)
        return fused, scores, indices

    def _run_source_encoded(
        self,
        source: str,
        query_tokens: torch.Tensor,
        query_vec: torch.Tensor,
        db_proj: nn.Linear,
        fusion: CrossAttentionFusion,
        retriever: LazyFaissRetriever,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores, indices, retrieved_keys, valid = self._search_index(
            query_vec=query_vec,
            retriever=retriever,
            top_k=self.top_k,
        )
        if self.retrieval_pipeline == "embed_project":
            retrieved, retrieved_masks = self._prepare_embed_project_retrieved(
                retrieved_keys=retrieved_keys,
                valid=valid,
                db_proj=db_proj,
            )
        elif self.retrieval_pipeline == "rawseq_esm1b":
            retrieved, retrieved_masks, scores = self._prepare_rawseq_retrieved(
                source=source,
                indices=indices,
                scores=scores,
                valid=valid,
                device=query_tokens.device,
            )
        else:
            raise ValueError(f"Unsupported retrieval_pipeline={self.retrieval_pipeline}")

        fused = fusion(query_tokens, retrieved, scores, retrieved_masks=retrieved_masks)
        return fused, scores, indices

    def forward(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        tensor_batch = {k: v for k, v in batch.items() if torch.is_tensor(v)}
        metadata_batch = {k: v for k, v in batch.items() if not torch.is_tensor(v)}

        query_tokens = tensor_batch["seq_embedding"][..., 0]  # [B, N, D]
        query_mask = tensor_batch.get("seq_mask", None)
        if query_mask is not None:
            query_mask = query_mask[..., 0]  # [B, N]

        use_seq, use_struct = self._enabled_sources()
        if self.retrieval_pipeline in {"embed_project", "rawseq_esm1b"}:
            raw_sequences = metadata_batch.get("raw_sequence")
            if raw_sequences is None:
                raise ValueError(
                    f"retrieval_pipeline={self.retrieval_pipeline} requires `raw_sequence` metadata in batches. "
                    "Use manifest/dataset mode (not packed shards) or include raw_sequence in packed records."
                )
            if not isinstance(raw_sequences, (list, tuple)):
                raise ValueError("Expected batch['raw_sequence'] to be a list/tuple of strings.")
            if len(raw_sequences) != query_tokens.shape[0]:
                raise ValueError(
                    f"raw_sequence batch size mismatch: got {len(raw_sequences)} sequences "
                    f"for batch size {query_tokens.shape[0]}"
                )
            seq_query_vecs = self.seq_query_encoder(raw_sequences) if (use_seq and self.seq_query_encoder is not None) else None
            struct_query_vecs = (
                self.struct_query_encoder(raw_sequences)
                if (use_struct and self.struct_query_encoder is not None)
                else None
            )
        else:
            seq_query_vecs = None
            struct_query_vecs = None

        fused_list = []
        seq_scores_list = []
        struct_scores_list = []
        seq_indices_list = []
        struct_indices_list = []

        for b in range(query_tokens.shape[0]):
            q_tok = query_tokens[b]
            q_msk = query_mask[b] if query_mask is not None else None

            seq_fused = None
            struct_fused = None
            seq_scores = torch.zeros(self.top_k, device=q_tok.device)
            struct_scores = torch.zeros(self.top_k, device=q_tok.device)
            seq_indices = torch.full((self.top_k,), -1, device=q_tok.device, dtype=torch.long)
            struct_indices = torch.full((self.top_k,), -1, device=q_tok.device, dtype=torch.long)

            if use_seq and self.seq_retriever is not None:
                if self.retrieval_pipeline == "legacy":
                    seq_fused, seq_scores, seq_indices = self._run_source_legacy(
                        q_tok, q_msk,
                        self.seq_query_proj,
                        self.seq_db_proj,
                        self.seq_fusion,
                        self.seq_retriever,
                    )
                else:
                    assert seq_query_vecs is not None
                    seq_fused, seq_scores, seq_indices = self._run_source_encoded(
                        source="seq",
                        query_tokens=q_tok,
                        query_vec=seq_query_vecs[b],
                        db_proj=self.seq_db_proj,
                        fusion=self.seq_fusion,
                        retriever=self.seq_retriever,
                    )
            if use_struct and self.struct_retriever is not None:
                if self.retrieval_pipeline == "legacy":
                    struct_fused, struct_scores, struct_indices = self._run_source_legacy(
                        q_tok, q_msk,
                        self.struct_query_proj,
                        self.struct_db_proj,
                        self.struct_fusion,
                        self.struct_retriever,
                    )
                else:
                    assert struct_query_vecs is not None
                    struct_fused, struct_scores, struct_indices = self._run_source_encoded(
                        source="struct",
                        query_tokens=q_tok,
                        query_vec=struct_query_vecs[b],
                        db_proj=self.struct_db_proj,
                        fusion=self.struct_fusion,
                        retriever=self.struct_retriever,
                    )

            if seq_fused is not None and struct_fused is not None:
                mix = torch.softmax(self.source_mix_logits, dim=0)
                fused = mix[0] * seq_fused + mix[1] * struct_fused
            elif seq_fused is not None:
                fused = seq_fused
            else:
                fused = struct_fused

            fused_list.append(fused)
            seq_scores_list.append(seq_scores)
            struct_scores_list.append(struct_scores)
            seq_indices_list.append(seq_indices)
            struct_indices_list.append(struct_indices)

        fused_batch = torch.stack(fused_list, dim=0)  # [B, N, D]
        num_recycles = tensor_batch["seq_embedding"].shape[-1]
        model_batch = dict(tensor_batch)
        model_batch["seq_embedding"] = fused_batch.unsqueeze(-1).expand(*fused_batch.shape, num_recycles)

        outputs = self.openfold(model_batch)
        outputs["seq_retrieval_scores"] = torch.stack(seq_scores_list, dim=0).detach()
        outputs["struct_retrieval_scores"] = torch.stack(struct_scores_list, dim=0).detach()
        outputs["seq_retrieval_indices"] = torch.stack(seq_indices_list, dim=0).detach()
        outputs["struct_retrieval_indices"] = torch.stack(struct_indices_list, dim=0).detach()
        outputs["retrieval_source_weights"] = torch.softmax(self.source_mix_logits, dim=0).detach()
        return outputs

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self(batch)
        tensor_batch = {k: v for k, v in batch.items() if torch.is_tensor(v)}
        labels = tensor_tree_map(lambda t: t[..., -1], tensor_batch)
        loss, breakdown = self.loss_fn(outputs, labels, _return_breakdown=True)

        for name, value in breakdown.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=False, logger=True)
        self.log(
            "train/seq_retrieval_entropy",
            -(outputs["seq_retrieval_scores"] * outputs["seq_retrieval_scores"].clamp(min=1e-8).log()).sum(dim=-1).mean(),
            on_step=True, on_epoch=False, logger=True,
        )
        self.log(
            "train/struct_retrieval_entropy",
            -(outputs["struct_retrieval_scores"] * outputs["struct_retrieval_scores"].clamp(min=1e-8).log()).sum(dim=-1).mean(),
            on_step=True, on_epoch=False, logger=True,
        )
        self.log("train/source_seq_weight", outputs["retrieval_source_weights"][0], on_step=True, on_epoch=False)
        self.log("train/source_struct_weight", outputs["retrieval_source_weights"][1], on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self(batch)
        tensor_batch = {k: v for k, v in batch.items() if torch.is_tensor(v)}
        labels = tensor_tree_map(lambda t: t[..., -1], tensor_batch)
        labels["use_clamped_fape"] = 0.0
        _, breakdown = self.loss_fn(outputs, labels, _return_breakdown=True)
        for name, value in breakdown.items():
            self.log(f"val/{name}", value, on_step=False, on_epoch=True, logger=True, sync_dist=False)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.lr, eps=1e-5)


def _auto_accelerator() -> str:
    return "gpu" if torch.cuda.is_available() else "cpu"


def _build_trainer_logger(args, output_dir: Path):
    """Build Lightning logger configuration.

    Returns:
        - `True` for Lightning's default logger behavior (CSV by default).
        - A `WandbLogger` instance when --use_wandb is enabled.
    """
    if not args.use_wandb:
        return True

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    elif args.wandb_api_key_env and args.wandb_api_key_env in os.environ:
        logger.info("Using W&B API key from env var %s", args.wandb_api_key_env)
    else:
        logger.warning(
            "W&B enabled but no API key provided via --wandb_api_key or env var %s. "
            "Proceeding with existing wandb login state.",
            args.wandb_api_key_env,
        )

    try:
        from pytorch_lightning.loggers import WandbLogger
    except Exception as exc:
        raise SystemExit(
            "W&B logging requested but WandbLogger is unavailable. Install wandb in the active environment."
        ) from exc

    kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
        "save_dir": str(output_dir),
        "offline": args.wandb_offline,
    }
    if args.wandb_run_id:
        kwargs["id"] = args.wandb_run_id
        kwargs["resume"] = args.wandb_resume

    if args.wandb_tags:
        kwargs["tags"] = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    wb_logger = WandbLogger(**kwargs)
    logger.info(
        "W&B logger enabled (project=%s, entity=%s, run_id=%s, offline=%s)",
        args.wandb_project,
        args.wandb_entity,
        args.wandb_run_id,
        args.wandb_offline,
    )
    return wb_logger


def main():
    parser = argparse.ArgumentParser(description="Train retrieval-augmented OpenFold with lazy FAISS retrieval")
    parser.add_argument(
        "--packed_dataset_dir",
        type=Path,
        default=None,
        help="Path to packed dataset generated by build_packed_dataset.py (preferred for speed)",
    )
    parser.add_argument("--dataset_dir", type=Path, default=None, help="Path to training_data directory")
    parser.add_argument("--manifest_path", type=Path, default=None, help="Path to JSONL manifest (optional)")
    parser.add_argument(
        "--write_manifest_to",
        type=Path,
        default=None,
        help="If set with --dataset_dir, writes manifest JSONL before training",
    )
    parser.add_argument("--seq_embedding_dir", type=Path, default=None, help="Directory of per-sequence embeddings")

    parser.add_argument("--struct_index_path", type=str, default=None, help="FAISS index path for structure retrieval")
    parser.add_argument("--seq_index_path", type=str, default=None, help="FAISS index path for sequence retrieval")
    parser.add_argument("--struct_index_dim", type=int, default=512)
    parser.add_argument("--seq_index_dim", type=int, default=1280)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument(
        "--retrieval_ablation",
        type=str,
        choices=["both", "seq_only", "struct_only"],
        default="struct_only",
    )
    parser.add_argument(
        "--retrieval_pipeline",
        type=str,
        choices=["legacy", "embed_project", "rawseq_esm1b"],
        default="legacy",
        help=(
            "legacy: pooled seq_embedding -> linear query projection (previous behavior); "
            "embed_project: explicit trainable ESM2/TMVec query encoders + reconstructed vectors projected to 1280; "
            "rawseq_esm1b: explicit trainable ESM2/TMVec query encoders + retrieved raw sequences embedded by frozen ESM1b."
        ),
    )
    parser.add_argument("--retriever_esm2_model_name", type=str, default="esm2_t12_35M_UR50D")
    parser.add_argument("--retriever_esm2_repr_layer", type=int, default=12)
    parser.add_argument("--retriever_esm2_max_len", type=int, default=1022)
    parser.add_argument("--retriever_tmvec_checkpoint", type=str, default=None)
    parser.add_argument("--retriever_tmvec_max_len", type=int, default=1022)
    parser.add_argument(
        "--retriever_normalize_queries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize ESM2/TMVec query vectors before FAISS retrieval scoring.",
    )
    parser.add_argument("--seq_index_ids_path", type=str, default=None, help="Row-aligned ids map for sequence FAISS index")
    parser.add_argument("--struct_index_ids_path", type=str, default=None, help="Row-aligned ids map for structure FAISS index")
    parser.add_argument("--seq_db_fasta_path", type=str, default=None, help="Sequence database FASTA for raw sequence lookup")
    parser.add_argument(
        "--seq_db_fasta_index_db",
        type=str,
        default=None,
        help="Biopython SeqIO sqlite index path for seq_db_fasta_path",
    )
    parser.add_argument("--struct_db_fasta_path", type=str, default=None, help="Structure database FASTA for raw sequence lookup")
    parser.add_argument(
        "--struct_db_fasta_index_db",
        type=str,
        default=None,
        help="Biopython SeqIO sqlite index path for struct_db_fasta_path",
    )
    parser.add_argument("--retrieved_esm1b_model_name", type=str, default="esm1b_t33_650M_UR50S")
    parser.add_argument("--retrieved_esm1b_repr_layer", type=int, default=33)
    parser.add_argument("--retrieved_esm1b_max_len", type=int, default=1022)
    parser.add_argument(
        "--retrieved_esm1b_device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for frozen ESM1b embedding of retrieved raw sequences (pipeline rawseq_esm1b only).",
    )
    parser.add_argument("--retrieved_esm1b_cache_size", type=int, default=4096)

    parser.add_argument("--config_preset", type=str, default="seqemb_initial_training")
    parser.add_argument("--openfold_checkpoint", type=str, default=None)
    parser.add_argument(
        "--train_openfold_all",
        action="store_true",
        help="Unfreeze all OpenFold modules (input/recycling/template/extra_msa/evoformer/structure/aux_heads).",
    )
    parser.add_argument("--train_input_embedder", action="store_true")
    parser.add_argument("--train_recycling_embedder", action="store_true")
    parser.add_argument("--train_template_embedder", action="store_true")
    parser.add_argument("--train_extra_msa_embedder", action="store_true")
    parser.add_argument("--train_extra_msa_stack", action="store_true")
    parser.add_argument("--train_evoformer", action="store_true")
    parser.add_argument(
        "--train_structure_module",
        action="store_true",
        help="Unfreeze OpenFold structure_module.",
    )
    parser.add_argument("--train_aux_heads", action="store_true")
    parser.add_argument("--seq_embedding_dim", type=int, default=1280)
    parser.add_argument("--fusion_heads", type=int, default=8)
    parser.add_argument("--fusion_dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--batch_size",
        "--batch_size_per_gpu",
        dest="batch_size",
        type=int,
        default=1,
        help="Per-device batch size (same as --batch_size_per_gpu).",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of optimizer accumulation steps (effective batch multiplier).",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=1000,
        help="Run validation every N training steps (default: 1000).",
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--max_recycling_iters", type=int, default=0)
    parser.add_argument("--strict_seq_embeddings", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("./outputs/retrieval_lightning"))
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="openfold-retrieval")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument(
        "--wandb_resume",
        type=str,
        choices=["allow", "must", "never", "auto"],
        default="allow",
        help="W&B resume policy when --wandb_run_id is provided",
    )
    parser.add_argument(
        "--wandb_api_key_env",
        type=str,
        default="WANDB_API_KEY",
        help="Environment variable containing W&B API key",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="Explicit W&B API key (prefer env var for security)",
    )
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated W&B tags")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    pl.seed_everything(args.seed, workers=True)
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be >= 1")
    if args.accumulate_grad_batches < 1:
        raise SystemExit("--accumulate_grad_batches must be >= 1")
    if args.devices < 1:
        raise SystemExit("--devices must be >= 1")
    if args.val_check_interval < 1:
        raise SystemExit("--val_check_interval must be >= 1")

    effective_batch_size = args.batch_size * args.accumulate_grad_batches * args.devices
    logger.info(
        "Batch config: per_device_batch_size=%d accumulate_grad_batches=%d devices=%d => effective_batch_size=%d",
        args.batch_size,
        args.accumulate_grad_batches,
        args.devices,
        effective_batch_size,
    )

    if args.retrieval_pipeline in {"embed_project", "rawseq_esm1b"} and args.packed_dataset_dir is not None:
        raise SystemExit(
            "--retrieval_pipeline embed_project/rawseq_esm1b requires raw sequences in each batch. "
            "Use --dataset_dir/--manifest_path (not --packed_dataset_dir), or extend packed shards to include raw_sequence."
        )

    if args.packed_dataset_dir is not None:
        data_module = RetrievalDataModule(
            packed_dataset_dir=args.packed_dataset_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    elif args.manifest_path is not None:
        data_module = RetrievalDataModule(
            manifest_path=args.manifest_path,
            seq_embedding_dir=args.seq_embedding_dir,
            config_preset=args.config_preset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_recycling_iters=args.max_recycling_iters,
            strict_seq_embeddings=args.strict_seq_embeddings,
            seq_embedding_dim=args.seq_embedding_dim,
        )
    else:
        if args.dataset_dir is None:
            raise SystemExit("Provide --packed_dataset_dir, --manifest_path, or --dataset_dir")
        if args.write_manifest_to is not None:
            records = build_manifest(args.dataset_dir, strict=True)
            write_manifest_jsonl(records, args.write_manifest_to)
            logger.info("Wrote manifest to %s", args.write_manifest_to)
        data_module = RetrievalDataModule(
            dataset_dir=args.dataset_dir,
            seq_embedding_dir=args.seq_embedding_dir,
            config_preset=args.config_preset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_recycling_iters=args.max_recycling_iters,
            strict_seq_embeddings=args.strict_seq_embeddings,
            seq_embedding_dim=args.seq_embedding_dim,
        )

    model = RetrievalAugmentedLightningModule(
        config_preset=args.config_preset,
        seq_embedding_dim=args.seq_embedding_dim,
        top_k=args.top_k,
        lr=args.lr,
        struct_index_path=args.struct_index_path,
        struct_index_dim=args.struct_index_dim,
        seq_index_path=args.seq_index_path,
        seq_index_dim=args.seq_index_dim,
        nprobe=args.nprobe,
        fusion_heads=args.fusion_heads,
        fusion_dropout=args.fusion_dropout,
        retrieval_ablation=args.retrieval_ablation,
        retrieval_pipeline=args.retrieval_pipeline,
        retriever_esm2_model_name=args.retriever_esm2_model_name,
        retriever_esm2_repr_layer=args.retriever_esm2_repr_layer,
        retriever_esm2_max_len=args.retriever_esm2_max_len,
        retriever_tmvec_checkpoint=args.retriever_tmvec_checkpoint,
        retriever_tmvec_max_len=args.retriever_tmvec_max_len,
        retriever_normalize_queries=args.retriever_normalize_queries,
        seq_index_ids_path=args.seq_index_ids_path,
        struct_index_ids_path=args.struct_index_ids_path,
        seq_db_fasta_path=args.seq_db_fasta_path,
        struct_db_fasta_path=args.struct_db_fasta_path,
        seq_db_fasta_index_db=args.seq_db_fasta_index_db,
        struct_db_fasta_index_db=args.struct_db_fasta_index_db,
        retrieved_esm1b_model_name=args.retrieved_esm1b_model_name,
        retrieved_esm1b_repr_layer=args.retrieved_esm1b_repr_layer,
        retrieved_esm1b_max_len=args.retrieved_esm1b_max_len,
        retrieved_esm1b_device=args.retrieved_esm1b_device,
        retrieved_esm1b_cache_size=args.retrieved_esm1b_cache_size,
        openfold_checkpoint=args.openfold_checkpoint,
        train_openfold_all=args.train_openfold_all,
        train_input_embedder=args.train_input_embedder,
        train_recycling_embedder=args.train_recycling_embedder,
        train_template_embedder=args.train_template_embedder,
        train_extra_msa_embedder=args.train_extra_msa_embedder,
        train_extra_msa_stack=args.train_extra_msa_stack,
        train_evoformer=args.train_evoformer,
        train_structure_module=args.train_structure_module,
        train_aux_heads=args.train_aux_heads,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer_logger = _build_trainer_logger(args, args.output_dir)
    trainer = pl.Trainer(
        default_root_dir=str(args.output_dir),
        logger=trainer_logger,
        accelerator=_auto_accelerator(),
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        val_check_interval=args.val_check_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
