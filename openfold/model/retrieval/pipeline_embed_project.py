from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interfaces import QueryPipeline, QueryVectors


def _sanitize_aa_sequence(seq: str) -> str:
    seq = str(seq).upper()
    return re.sub(r"[^A-Z]", "X", seq)


class ESM2QueryEncoder(nn.Module):
    """Trainable ESM2 query encoder for sequence-side retrieval."""

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
        self.batch_converter = alphabet.get_batch_converter(truncation_seq_length=truncation_seq_length)
        self.repr_layer = int(repr_layer)
        self.normalize = bool(normalize)

    def forward(self, sequences: Sequence[str]) -> torch.Tensor:
        if not sequences:
            device = next(self.model.parameters()).device
            dim = int(getattr(self.model, "embed_dim", 480))
            return torch.empty(0, dim, device=device)

        batch = [(str(i), _sanitize_aa_sequence(s)) for i, s in enumerate(sequences)]
        _, _, toks = self.batch_converter(batch)
        device = next(self.model.parameters()).device
        toks = toks.to(device=device, non_blocking=True)

        out = self.model(toks, repr_layers=[self.repr_layer], return_contacts=False)
        reps = out["representations"][self.repr_layer]
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
    """Trainable TMVec-2s query encoder for structure-side retrieval."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        max_length: int = 1022,
        normalize: bool = True,
    ):
        super().__init__()
        repo_root = Path(__file__).resolve().parents[3]
        tmvec_root = repo_root / "tmvec-bench"
        if str(tmvec_root) not in sys.path:
            sys.path.insert(0, str(tmvec_root))

        try:
            from src.model.tmvec2_student_model import StudentModel, encode_sequence
        except Exception as exc:
            raise RuntimeError(
                "Failed to import TMVec-2s model from tmvec-bench."
            ) from exc

        self._encode_sequence = encode_sequence
        self.model = StudentModel()
        self.max_length = int(max_length)
        self.normalize = bool(normalize)

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            else:
                state_dict = ckpt
            self.model.load_state_dict(state_dict, strict=True)

    def forward(self, sequences: Sequence[str]) -> torch.Tensor:
        if not sequences:
            device = next(self.model.parameters()).device
            return torch.empty(0, 512, device=device)

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


class EmbedProjectQueryPipeline(QueryPipeline):
    """Explicit trainable query encoders for retrieval sources."""

    def __init__(
        self,
        use_seq_encoder: bool = True,
        use_struct_encoder: bool = True,
        retriever_esm2_model_name: str = "esm2_t12_35M_UR50D",
        retriever_esm2_repr_layer: int = 12,
        retriever_esm2_max_len: int = 1022,
        retriever_tmvec_checkpoint: Optional[str] = None,
        retriever_tmvec_max_len: int = 1022,
        retriever_normalize_queries: bool = True,
        seq_encoder_override: Optional[nn.Module] = None,
        struct_encoder_override: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.seq_encoder = seq_encoder_override
        self.struct_encoder = struct_encoder_override

        if self.seq_encoder is None and use_seq_encoder:
            self.seq_encoder = ESM2QueryEncoder(
                model_name=retriever_esm2_model_name,
                repr_layer=retriever_esm2_repr_layer,
                truncation_seq_length=retriever_esm2_max_len,
                normalize=retriever_normalize_queries,
            )
        if self.struct_encoder is None and use_struct_encoder:
            self.struct_encoder = TMVec2QueryEncoder(
                checkpoint_path=retriever_tmvec_checkpoint,
                max_length=retriever_tmvec_max_len,
                normalize=retriever_normalize_queries,
            )

    def encode_queries(
        self,
        query_tokens: torch.Tensor,
        query_mask: Optional[torch.Tensor],
        raw_sequences: Optional[Sequence[str]],
        use_seq: bool,
        use_struct: bool,
    ) -> QueryVectors:
        del query_tokens, query_mask
        if raw_sequences is None:
            raise ValueError(
                "embed_project pipeline requires raw_sequence metadata in each batch"
            )
        if not isinstance(raw_sequences, (list, tuple)):
            raise ValueError("Expected raw_sequences to be a list/tuple of strings")

        seq = None
        struct = None
        if use_seq:
            if self.seq_encoder is None:
                raise RuntimeError("Sequence query encoder is not configured")
            seq = self.seq_encoder(raw_sequences)
        if use_struct:
            if self.struct_encoder is None:
                raise RuntimeError("Structure query encoder is not configured")
            struct = self.struct_encoder(raw_sequences)

        return QueryVectors(seq=seq, struct=struct)
