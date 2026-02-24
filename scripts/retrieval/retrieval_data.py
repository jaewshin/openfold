#!/usr/bin/env python3
"""Utilities for parsing retrieval training fixtures and building dataloaders.

This module is intentionally lightweight and memory-aware:
- The manifest only stores metadata and paths.
- Per-example features are built lazily from mmCIF + sequence.
- Sequence embeddings can be loaded from disk, with a deterministic fallback
  to avoid loading large sequence encoders during experimentation.
"""

from __future__ import annotations

import gzip
import json
import logging
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from openfold.config import model_config
from openfold.data import data_pipeline, feature_pipeline, mmcif_parsing
from openfold.np import protein, residue_constants

logger = logging.getLogger(__name__)


def parse_fasta(path: Path) -> Dict[str, str]:
    records: Dict[str, str] = {}
    header: Optional[str] = None
    chunks: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(chunks)
                header = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        records[header] = "".join(chunks)
    return records


@dataclass(frozen=True)
class TrainingRecord:
    sequence_id: str
    split: str
    pdb_id: str
    chain_id: str
    sequence: str
    structure_path: str
    domain_id: str = ""
    cath_code: str = ""
    topology_code: str = ""
    superfamily_code: str = ""
    n_residues: int = 0

    def to_json(self) -> Dict[str, object]:
        return {
            "sequence_id": self.sequence_id,
            "split": self.split,
            "pdb_id": self.pdb_id,
            "chain_id": self.chain_id,
            "sequence": self.sequence,
            "structure_path": self.structure_path,
            "domain_id": self.domain_id,
            "cath_code": self.cath_code,
            "topology_code": self.topology_code,
            "superfamily_code": self.superfamily_code,
            "n_residues": self.n_residues,
        }

    @staticmethod
    def from_json(obj: Dict[str, object]) -> "TrainingRecord":
        return TrainingRecord(
            sequence_id=str(obj["sequence_id"]),
            split=str(obj["split"]),
            pdb_id=str(obj["pdb_id"]),
            chain_id=str(obj["chain_id"]),
            sequence=str(obj["sequence"]),
            structure_path=str(obj["structure_path"]),
            domain_id=str(obj.get("domain_id", "")),
            cath_code=str(obj.get("cath_code", "")),
            topology_code=str(obj.get("topology_code", "")),
            superfamily_code=str(obj.get("superfamily_code", "")),
            n_residues=int(obj.get("n_residues", 0)),
        )


def _find_structure_path(structures_dir: Path, pdb_id: str) -> Optional[Path]:
    pdb_id = pdb_id.lower()
    for suffix in (".cif.gz", ".cif", ".pdb.gz", ".pdb"):
        p = structures_dir / f"{pdb_id}{suffix}"
        if p.exists():
            return p
    return None


def build_manifest(dataset_dir: Path, strict: bool = True) -> List[TrainingRecord]:
    """Build a normalized manifest from download_structures.py outputs."""
    splits_path = dataset_dir / "splits.json"
    train_fasta = dataset_dir / "train.fasta"
    val_fasta = dataset_dir / "val.fasta"
    structures_dir = dataset_dir / "structures"

    for p in (splits_path, train_fasta, val_fasta, structures_dir):
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    split_rows = json.loads(splits_path.read_text())
    if not isinstance(split_rows, list):
        raise ValueError(f"Expected list-style splits.json, got {type(split_rows).__name__}")

    train_map = parse_fasta(train_fasta)
    val_map = parse_fasta(val_fasta)

    records: List[TrainingRecord] = []
    missing_seq = 0
    missing_struct = 0

    for row in split_rows:
        split = str(row.get("split", ""))
        if split not in {"train", "val"}:
            continue

        pdb_id = str(row["pdb_id"]).lower()
        chain_id = str(row["chain_id"])
        seq_id = f"{pdb_id}_{chain_id}"
        seq = train_map.get(seq_id) if split == "train" else val_map.get(seq_id)

        if not seq:
            missing_seq += 1
            if strict:
                raise ValueError(f"Missing sequence for {seq_id} in {split}.fasta")
            continue

        struct_path = _find_structure_path(structures_dir, pdb_id)
        if struct_path is None:
            missing_struct += 1
            if strict:
                raise ValueError(f"Missing structure file for pdb_id={pdb_id}")
            continue

        records.append(
            TrainingRecord(
                sequence_id=seq_id,
                split=split,
                pdb_id=pdb_id,
                chain_id=chain_id,
                sequence=seq,
                structure_path=str(struct_path),
                domain_id=str(row.get("domain_id", "")),
                cath_code=str(row.get("cath_code", "")),
                topology_code=str(row.get("topology_code", "")),
                superfamily_code=str(row.get("superfamily_code", "")),
                n_residues=int(row.get("n_residues", len(seq))),
            )
        )

    if missing_seq or missing_struct:
        logger.warning(
            "Manifest built with skips: missing_seq=%d missing_struct=%d", missing_seq, missing_struct
        )

    records.sort(key=lambda r: (r.split, r.sequence_id))
    return records


def write_manifest_jsonl(records: Sequence[TrainingRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r.to_json()) + "\n")


def read_manifest_jsonl(path: Path) -> List[TrainingRecord]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(TrainingRecord.from_json(json.loads(line)))
    return records


class _LRUCache:
    """Small LRU cache to avoid repeated mmCIF parsing within an epoch."""

    def __init__(self, max_size: int = 32):
        self.max_size = max_size
        self._store: "OrderedDict[str, object]" = OrderedDict()

    def get(self, key: str):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, value) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)


class RetrievalStructureDataset(Dataset):
    """Dataset that emits OpenFold features for retrieval-augmented training."""

    def __init__(
        self,
        records: Sequence[TrainingRecord],
        config_preset: str = "seqemb_initial_training",
        mode: str = "train",
        seq_embedding_dir: Optional[Path] = None,
        seq_embedding_dim: int = 1280,
        strict_seq_embeddings: bool = False,
        low_prec: bool = False,
        disable_templates: bool = True,
        max_recycling_iters: int = 0,
        mmcif_cache_size: int = 32,
    ):
        super().__init__()
        self.records = list(records)
        self.mode = mode
        self.seq_embedding_dir = Path(seq_embedding_dir) if seq_embedding_dir else None
        self.seq_embedding_dim = seq_embedding_dim
        self.strict_seq_embeddings = strict_seq_embeddings
        self._mmcif_cache = _LRUCache(max_size=mmcif_cache_size)

        self.config = model_config(config_preset, train=(mode == "train"), low_prec=low_prec)
        self.config.data.common.max_recycling_iters = max_recycling_iters
        if disable_templates:
            self.config.model.template.enabled = False
            self.config.data.common.use_templates = False

        self.feature_pipeline = feature_pipeline.FeaturePipeline(self.config.data)
        rng = np.random.default_rng(0)
        self._fallback_aa_table = rng.standard_normal((21, seq_embedding_dim)).astype(np.float32) * 0.05
        self._aa_vocab = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        self._invalid_warn_count = 0
        self._invalid_warn_limit = 16

    def __len__(self) -> int:
        return len(self.records)

    def _read_mmcif_object(self, record: TrainingRecord):
        cached = self._mmcif_cache.get(record.structure_path)
        if cached is not None:
            return cached

        path = Path(record.structure_path)
        if path.suffix == ".gz":
            mmcif_string = gzip.open(path, "rt").read()
        else:
            mmcif_string = path.read_text()

        parsed = mmcif_parsing.parse(file_id=record.pdb_id, mmcif_string=mmcif_string)
        if parsed.mmcif_object is None:
            raise ValueError(f"Failed to parse mmCIF: {record.structure_path} ({parsed.errors})")

        mm_obj = parsed.mmcif_object
        self._mmcif_cache.put(record.structure_path, mm_obj)
        return mm_obj

    @staticmethod
    def _is_pdb_path(path: Path) -> bool:
        suffixes = {s.lower() for s in path.suffixes}
        return ".pdb" in suffixes or ".ent" in suffixes

    @staticmethod
    def _protein_to_sequence(protein_object: protein.Protein) -> str:
        return "".join(residue_constants.restypes_with_x[int(i)] for i in protein_object.aatype)

    @staticmethod
    def _resize_embedding_length(emb: np.ndarray, target_len: int) -> np.ndarray:
        if emb.shape[0] == target_len:
            return emb
        if emb.shape[0] > target_len:
            return emb[:target_len]
        pad = np.zeros((target_len - emb.shape[0], emb.shape[1]), dtype=np.float32)
        return np.concatenate([emb, pad], axis=0)

    def _read_pdb_object(self, record: TrainingRecord) -> protein.Protein:
        path = Path(record.structure_path)
        if path.suffix == ".gz":
            pdb_string = gzip.open(path, "rt").read()
        else:
            pdb_string = path.read_text()

        # Most OpenProteinNet PDB fixtures are single-chain; parse all chains by default.
        chain_id = record.chain_id if record.chain_id else None
        try:
            return protein.from_pdb_string(pdb_string, chain_id=chain_id)
        except Exception:
            return protein.from_pdb_string(pdb_string, chain_id=None)

    def _fallback_seq_embedding(self, sequence: str) -> np.ndarray:
        idxs = [self._aa_vocab.get(ch, 20) for ch in sequence]
        return self._fallback_aa_table[np.array(idxs, dtype=np.int64)]

    def _load_seq_embedding(self, record: TrainingRecord) -> np.ndarray:
        if self.seq_embedding_dir is None:
            return self._fallback_seq_embedding(record.sequence)

        base = self.seq_embedding_dir / record.sequence_id
        candidates = [base.with_suffix(".npy"), base.with_suffix(".pt")]
        for p in candidates:
            if not p.exists():
                continue
            if p.suffix == ".npy":
                emb = np.load(p).astype(np.float32, copy=False)
            else:
                obj = torch.load(p, map_location="cpu", weights_only=False)
                if isinstance(obj, dict) and "representations" in obj:
                    emb = obj["representations"][33].cpu().numpy().astype(np.float32, copy=False)
                elif torch.is_tensor(obj):
                    emb = obj.cpu().numpy().astype(np.float32, copy=False)
                else:
                    raise ValueError(f"Unsupported embedding .pt format: {p}")
            if emb.shape[0] != len(record.sequence):
                # Keep shape consistent with the sequence by truncation/padding.
                n = len(record.sequence)
                if emb.shape[0] > n:
                    emb = emb[:n]
                else:
                    pad = np.zeros((n - emb.shape[0], emb.shape[1]), dtype=np.float32)
                    emb = np.concatenate([emb, pad], axis=0)
            return emb

        if self.strict_seq_embeddings:
            raise FileNotFoundError(
                f"Missing embedding file for {record.sequence_id} in {self.seq_embedding_dir}"
            )
        return self._fallback_seq_embedding(record.sequence)

    def _validate_processed_features(
        self, record: TrainingRecord, feats: Dict[str, object]
    ) -> Optional[str]:
        all_atom_mask = feats.get("all_atom_mask")
        all_atom_positions = feats.get("all_atom_positions")
        if not torch.is_tensor(all_atom_mask) or not torch.is_tensor(all_atom_positions):
            return "missing_all_atom_tensors"
        if not torch.isfinite(all_atom_mask).all():
            return "non_finite_all_atom_mask"
        if not torch.isfinite(all_atom_positions).all():
            return "non_finite_all_atom_positions"

        mask_sum = float(all_atom_mask.sum().item())
        if mask_sum <= 0.0:
            return "empty_all_atom_mask"

        ca_pos = residue_constants.atom_order["CA"]
        if all_atom_mask.ndim >= 3:
            ca_mask = all_atom_mask[..., ca_pos, :]
        else:
            ca_mask = all_atom_mask[..., ca_pos]
        ca_sum = float(ca_mask.sum().item())
        if ca_sum <= 0.0:
            return "empty_ca_mask"

        return None

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        record = self.records[idx]
        path = Path(record.structure_path)

        if self._is_pdb_path(path):
            pdb_obj = self._read_pdb_object(record)
            struct_sequence = self._protein_to_sequence(pdb_obj)
            struct_feats = data_pipeline.make_pdb_features(
                pdb_obj,
                description=record.sequence_id,
                is_distillation=True,
            )
        else:
            mm_obj = self._read_mmcif_object(record)
            if record.chain_id not in mm_obj.chain_to_seqres:
                raise KeyError(
                    f"Chain {record.chain_id} not found in mmCIF for {record.sequence_id} "
                    f"({record.structure_path})"
                )
            struct_sequence = mm_obj.chain_to_seqres[record.chain_id]
            struct_feats = data_pipeline.make_mmcif_features(mm_obj, record.chain_id)

        msa_feats = data_pipeline.make_dummy_msa_feats(struct_sequence)
        seq_emb = self._resize_embedding_length(self._load_seq_embedding(record), len(struct_sequence))

        raw = {
            **struct_feats,
            **msa_feats,
            "seq_embedding": seq_emb,
        }
        feats = self.feature_pipeline.process_features(raw, mode=self.mode)
        # Keep query sequence metadata for retrieval encoders that operate on raw amino-acid strings.
        feats["raw_sequence"] = record.sequence
        feats["sequence_id"] = record.sequence_id

        invalid_reason = self._validate_processed_features(record, feats)
        if invalid_reason is not None:
            if self._invalid_warn_count < self._invalid_warn_limit:
                logger.warning(
                    "Skipping invalid sample %s (split=%s reason=%s path=%s)",
                    record.sequence_id,
                    record.split,
                    invalid_reason,
                    record.structure_path,
                )
            self._invalid_warn_count += 1
            return None

        return feats


def collate_feature_dicts(
    samples: Sequence[Optional[Dict[str, object]]]
) -> Optional[Dict[str, object]]:
    """Collate tensors by stacking and keep metadata as lists."""
    samples = [s for s in samples if s is not None]
    if not samples:
        return None
    out: Dict[str, object] = {}
    keys = samples[0].keys()
    for k in keys:
        v0 = samples[0][k]
        if torch.is_tensor(v0):
            out[k] = torch.stack([s[k] for s in samples], dim=0)
        else:
            out[k] = [s[k] for s in samples]
    return out


def _flush_packed_shard(
    shard_out_path: Path,
    chunk: Sequence[Dict[str, torch.Tensor]],
    sequence_ids: Sequence[str],
    float16_storage: bool,
) -> Dict[str, object]:
    if not chunk:
        raise ValueError("Cannot flush empty shard chunk")

    tensor_keys = sorted([k for k, v in chunk[0].items() if torch.is_tensor(v)])
    features: Dict[str, torch.Tensor] = {}
    for k in tensor_keys:
        stacked = torch.stack([sample[k].cpu() for sample in chunk], dim=0).contiguous()
        if float16_storage and stacked.dtype == torch.float32:
            stacked = stacked.to(torch.float16)
        features[k] = stacked

    torch.save({"features": features, "sequence_ids": list(sequence_ids)}, shard_out_path)
    return {
        "path": shard_out_path.name,
        "size": len(sequence_ids),
        "sequence_ids": list(sequence_ids),
    }


def pack_dataset_split(
    dataset: Dataset,
    records: Sequence[TrainingRecord],
    output_dir: Path,
    shard_size: int = 64,
    float16_storage: bool = False,
) -> Dict[str, object]:
    """Serialize one dataset split into sharded tensor files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    shards: List[Dict[str, object]] = []
    chunk: List[Dict[str, torch.Tensor]] = []
    chunk_ids: List[str] = []
    total = 0
    skipped = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        if sample is None:
            skipped += 1
            continue
        tensor_sample = {k: v for k, v in sample.items() if torch.is_tensor(v)}
        chunk.append(tensor_sample)
        chunk_ids.append(records[i].sequence_id)
        if len(chunk) >= shard_size:
            shard_path = output_dir / f"shard_{len(shards):05d}.pt"
            shards.append(_flush_packed_shard(shard_path, chunk, chunk_ids, float16_storage=float16_storage))
            total += len(chunk)
            chunk.clear()
            chunk_ids.clear()

    if chunk:
        shard_path = output_dir / f"shard_{len(shards):05d}.pt"
        shards.append(_flush_packed_shard(shard_path, chunk, chunk_ids, float16_storage=float16_storage))
        total += len(chunk)

    metadata = {
        "num_samples": total,
        "num_skipped": skipped,
        "num_shards": len(shards),
        "shards": shards,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def build_packed_feature_dataset(
    output_dir: Path,
    dataset_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    seq_embedding_dir: Optional[Path] = None,
    config_preset: str = "seqemb_initial_training",
    low_prec: bool = False,
    disable_templates: bool = True,
    max_recycling_iters: int = 0,
    strict_seq_embeddings: bool = False,
    seq_embedding_dim: int = 1280,
    shard_size: int = 64,
    float16_storage: bool = False,
) -> Dict[str, object]:
    """Precompute OpenFold-ready features into a packed sharded dataset."""
    if dataset_dir is None and manifest_path is None:
        raise ValueError("Provide either dataset_dir or manifest_path")

    if manifest_path is not None:
        records = read_manifest_jsonl(manifest_path)
    else:
        records = build_manifest(dataset_dir, strict=True)

    train_records = [r for r in records if r.split == "train"]
    val_records = [r for r in records if r.split == "val"]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = RetrievalStructureDataset(
        records=train_records,
        config_preset=config_preset,
        mode="train",
        seq_embedding_dir=seq_embedding_dir,
        seq_embedding_dim=seq_embedding_dim,
        strict_seq_embeddings=strict_seq_embeddings,
        low_prec=low_prec,
        disable_templates=disable_templates,
        max_recycling_iters=max_recycling_iters,
    )
    val_dataset = RetrievalStructureDataset(
        records=val_records,
        config_preset=config_preset,
        mode="eval",
        seq_embedding_dir=seq_embedding_dir,
        seq_embedding_dim=seq_embedding_dim,
        strict_seq_embeddings=strict_seq_embeddings,
        low_prec=low_prec,
        disable_templates=disable_templates,
        max_recycling_iters=max_recycling_iters,
    )

    logger.info("Packing train split (%d samples)...", len(train_dataset))
    train_meta = pack_dataset_split(
        train_dataset,
        train_records,
        output_dir=output_dir / "train",
        shard_size=shard_size,
        float16_storage=float16_storage,
    )
    logger.info("Packing val split (%d samples)...", len(val_dataset))
    val_meta = pack_dataset_split(
        val_dataset,
        val_records,
        output_dir=output_dir / "val",
        shard_size=shard_size,
        float16_storage=float16_storage,
    )

    root_meta = {
        "format": "retrieval_packed_v1",
        "config_preset": config_preset,
        "seq_embedding_dim": seq_embedding_dim,
        "shard_size": shard_size,
        "float16_storage": float16_storage,
        "train": {"num_samples": train_meta["num_samples"], "num_shards": train_meta["num_shards"]},
        "val": {"num_samples": val_meta["num_samples"], "num_shards": val_meta["num_shards"]},
    }
    (output_dir / "metadata.json").write_text(json.dumps(root_meta, indent=2))
    return root_meta


class PackedRetrievalDataset(Dataset):
    """Dataset over precomputed packed feature shards."""

    def __init__(self, packed_dataset_dir: Path, split: str, shard_cache_size: int = 2):
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split={split!r}")

        self.split = split
        self.split_dir = Path(packed_dataset_dir) / split
        meta_path = self.split_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Packed split metadata not found: {meta_path}")

        self.meta = json.loads(meta_path.read_text())
        self.shards = self.meta["shards"]
        self.cum_counts: List[int] = []
        total = 0
        for s in self.shards:
            total += int(s["size"])
            self.cum_counts.append(total)
        self.total = total
        self._shard_cache = _LRUCache(max_size=shard_cache_size)

    def __len__(self):
        return self.total

    def _load_shard(self, shard_idx: int) -> Dict[str, object]:
        shard_path = str(self.split_dir / self.shards[shard_idx]["path"])
        cached = self._shard_cache.get(shard_path)
        if cached is not None:
            return cached
        data = torch.load(shard_path, map_location="cpu", weights_only=False)
        self._shard_cache.put(shard_path, data)
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)

        shard_idx = bisect_right(self.cum_counts, idx)
        start = 0 if shard_idx == 0 else self.cum_counts[shard_idx - 1]
        row = idx - start
        shard = self._load_shard(shard_idx)

        feats = {}
        for k, t in shard["features"].items():
            x = t[row]
            if x.dtype == torch.float16:
                x = x.float()
            feats[k] = x
        return feats


class RetrievalDataModule(pl.LightningDataModule):
    """Lightning DataModule for retrieval training data generated by download_structures.py."""

    def __init__(
        self,
        packed_dataset_dir: Optional[Path] = None,
        dataset_dir: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
        seq_embedding_dir: Optional[Path] = None,
        config_preset: str = "seqemb_initial_training",
        batch_size: int = 1,
        num_workers: int = 0,
        low_prec: bool = False,
        disable_templates: bool = True,
        max_recycling_iters: int = 0,
        strict_seq_embeddings: bool = False,
        seq_embedding_dim: int = 1280,
    ):
        super().__init__()
        if packed_dataset_dir is None and dataset_dir is None and manifest_path is None:
            raise ValueError("Provide packed_dataset_dir, dataset_dir, or manifest_path")

        self.packed_dataset_dir = Path(packed_dataset_dir) if packed_dataset_dir else None
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.seq_embedding_dir = Path(seq_embedding_dir) if seq_embedding_dir else None
        self.config_preset = config_preset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.low_prec = low_prec
        self.disable_templates = disable_templates
        self.max_recycling_iters = max_recycling_iters
        self.strict_seq_embeddings = strict_seq_embeddings
        self.seq_embedding_dim = seq_embedding_dim

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.packed_dataset_dir is not None:
            self.train_dataset = PackedRetrievalDataset(self.packed_dataset_dir, split="train")
            self.val_dataset = PackedRetrievalDataset(self.packed_dataset_dir, split="val")
            return

        if self.manifest_path is not None:
            records = read_manifest_jsonl(self.manifest_path)
        else:
            records = build_manifest(self.dataset_dir, strict=True)

        train_records = [r for r in records if r.split == "train"]
        val_records = [r for r in records if r.split == "val"]

        self.train_dataset = RetrievalStructureDataset(
            records=train_records,
            config_preset=self.config_preset,
            mode="train",
            seq_embedding_dir=self.seq_embedding_dir,
            seq_embedding_dim=self.seq_embedding_dim,
            strict_seq_embeddings=self.strict_seq_embeddings,
            low_prec=self.low_prec,
            disable_templates=self.disable_templates,
            max_recycling_iters=self.max_recycling_iters,
        )
        self.val_dataset = RetrievalStructureDataset(
            records=val_records,
            config_preset=self.config_preset,
            mode="eval",
            seq_embedding_dir=self.seq_embedding_dir,
            seq_embedding_dim=self.seq_embedding_dim,
            strict_seq_embeddings=self.strict_seq_embeddings,
            low_prec=self.low_prec,
            disable_templates=self.disable_templates,
            max_recycling_iters=self.max_recycling_iters,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_feature_dicts,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_feature_dicts,
            pin_memory=False,
        )
