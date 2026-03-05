from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from .lora import apply_lora_to_linear_modules, count_trainable_parameters

logger = logging.getLogger(__name__)


def _sanitize_aa_sequence(seq: str) -> str:
    seq = str(seq).upper()
    return re.sub(r"[^A-Z]", "X", seq)


def _parse_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(
        f"Unsupported esm_efficient_compute_dtype={name!r}. "
        "Expected one of {'bfloat16', 'float16', 'float32'}."
    )


def _map_esm_efficient_model_name(model_name: str) -> str:
    name = str(model_name).strip()
    if not name:
        return "esm1b"
    if os.path.exists(name):
        return name
    lower = name.lower()
    if lower in {"esm1b_t33_650m_ur50s", "esm1b"}:
        return "esm1b"
    return name


class ESM1bContextEncoder(nn.Module):
    """Token-level ESM1b encoder for retrieved raw-sequence context."""

    def __init__(
        self,
        model_name: str = "esm1b_t33_650M_UR50S",
        repr_layer: int = 33,
        truncation_seq_length: int = 1022,
        tuning_mode: str = "lora",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_modules: Sequence[str] = (
            "self_attn.q_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "fc1",
            "fc2",
        ),
        train_layer_norm: bool = False,
        backend: str = "fair_esm",
        esm_efficient_use_pretrained: bool = True,
        esm_efficient_compute_dtype: str = "bfloat16",
    ):
        super().__init__()

        backend_key = str(backend).strip().lower()
        if backend_key not in {"fair_esm", "esm_efficient"}:
            raise ValueError(
                f"Unsupported ESM1b backend={backend!r}. "
                "Expected one of {'fair_esm', 'esm_efficient'}."
            )
        self.backend = backend_key

        if self.backend == "fair_esm":
            self._init_fair_esm_model(
                model_name=model_name,
                truncation_seq_length=truncation_seq_length,
                repr_layer=repr_layer,
            )
        else:
            self._init_esm_efficient_model(
                model_name=model_name,
                repr_layer=repr_layer,
                use_pretrained=bool(esm_efficient_use_pretrained),
                compute_dtype=str(esm_efficient_compute_dtype),
            )

        requested_mode = str(tuning_mode).strip().lower()
        if requested_mode not in {"lora", "full", "frozen"}:
            raise ValueError(
                f"Unsupported ESM1b tuning_mode={tuning_mode!r}. "
                "Expected one of {'lora', 'full', 'frozen'}."
            )
        self.requested_tuning_mode = requested_mode
        self.effective_tuning_mode = requested_mode
        self._patched_lora_modules: Tuple[str, ...] = ()
        self._configure_tuning(
            tuning_mode=requested_mode,
            lora_rank=int(lora_rank),
            lora_alpha=float(lora_alpha),
            lora_dropout=float(lora_dropout),
            lora_target_modules=tuple(str(t) for t in lora_target_modules),
            train_layer_norm=bool(train_layer_norm),
        )

    def _init_fair_esm_model(
        self,
        model_name: str,
        truncation_seq_length: int,
        repr_layer: int,
    ) -> None:
        try:
            from esm import pretrained
        except Exception as exc:
            raise RuntimeError(
                "Failed to import `esm`. Install fair-esm in the active environment."
            ) from exc

        model, alphabet = pretrained.load_model_and_alphabet(model_name)
        self.model = model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter(truncation_seq_length=int(truncation_seq_length))
        self.repr_layer = int(repr_layer)
        self.embed_dim = int(getattr(model, "embed_dim", 1280))
        self._esme_tokenize = None
        self._esme_alphabet = None

    def _init_esm_efficient_model(
        self,
        model_name: str,
        repr_layer: int,
        use_pretrained: bool,
        compute_dtype: str,
    ) -> None:
        if int(repr_layer) != 33:
            raise ValueError(
                "ESM-efficient backend currently returns final-layer ESM1b representations only; "
                "set rawseq_esm1b.esm1b_repr_layer=33."
            )
        try:
            from esme import ESM, ESM1b
            from esme.alphabet import Alphabet, tokenize
        except Exception as exc:
            raise RuntimeError(
                "Failed to import `esme` (esm-efficient). Install esm-efficient in the active environment."
            ) from exc

        resolved_name = _map_esm_efficient_model_name(model_name)
        if use_pretrained:
            model = ESM.from_pretrained(resolved_name, device="cpu")
        else:
            model = ESM1b()

        dtype = _parse_dtype(compute_dtype)
        model = model.to(dtype=dtype)

        self.model = model
        self.alphabet = None
        self.batch_converter = None
        self.repr_layer = 33
        self.embed_dim = int(getattr(model, "embed_dim", 1280))
        self._esme_tokenize = tokenize
        self._esme_alphabet = Alphabet

    @staticmethod
    def _resolve_esm_efficient_lora_layers(targets: Sequence[str]) -> Tuple[str, ...]:
        layers = set()
        for raw in targets:
            token = str(raw).strip().lower()
            if not token:
                continue
            if ("q_proj" in token) or (".q" in token) or ("query" in token):
                layers.add("query")
            if ("k_proj" in token) or (".k" in token) or ("key" in token):
                layers.add("key")
            if ("v_proj" in token) or (".v" in token) or ("value" in token):
                layers.add("value")
            if ("out_proj" in token) or (".out" in token) or ("output" in token):
                layers.add("output")

        if not layers:
            layers = {"query", "value", "output"}
        return tuple(sorted(layers))

    def _configure_tuning(
        self,
        tuning_mode: str,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        lora_target_modules: Sequence[str],
        train_layer_norm: bool,
    ) -> None:
        if tuning_mode == "frozen":
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()
            return

        if tuning_mode == "full":
            for p in self.model.parameters():
                p.requires_grad_(True)
            self.model.train()
            return

        # LoRA mode.
        for p in self.model.parameters():
            p.requires_grad_(False)

        patched: List[str] = []
        if self.backend == "esm_efficient" and hasattr(self.model, "add_lora"):
            layers = self._resolve_esm_efficient_lora_layers(lora_target_modules)
            try:
                self.model.add_lora(
                    rank=int(lora_rank),
                    alpha=float(lora_alpha),
                    layers=layers,
                    dropout_p=float(lora_dropout),
                    adapter_names=["default"],
                )
                patched = [f"native::{name}" for name in layers]
            except Exception:
                logger.exception(
                    "Failed to apply native esm-efficient LoRA; "
                    "falling back to generic linear-module LoRA patching."
                )

        if not patched:
            patched = apply_lora_to_linear_modules(
                self.model,
                target_substrings=lora_target_modules,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
        self._patched_lora_modules = tuple(sorted(patched))

        if not patched:
            logger.warning(
                "ESM1b LoRA mode requested but no target modules matched %s; "
                "falling back to full finetuning.",
                list(lora_target_modules),
            )
            for p in self.model.parameters():
                p.requires_grad_(True)
            self.effective_tuning_mode = "full_fallback"
            self.model.train()
            return

        if train_layer_norm:
            for module in self.model.modules():
                if isinstance(module, nn.LayerNorm):
                    for p in module.parameters():
                        p.requires_grad_(True)

        self.model.train()

    def lora_trainable_params(self) -> int:
        return int(
            sum(
                p.numel()
                for name, p in self.model.named_parameters()
                if p.requires_grad and ("lora_A" in name or "lora_B" in name)
            )
        )

    def trainable_params(self) -> int:
        return count_trainable_parameters(self.model)

    def encode_with_ids(self, seq_pairs: Sequence[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        if not seq_pairs:
            return {}
        seq_ids = [sid for sid, _ in seq_pairs]
        seqs = [seq for _, seq in seq_pairs]
        embeddings = self(seqs)
        return {sid: emb for sid, emb in zip(seq_ids, embeddings)}

    def _forward_fair_esm(self, sequences: Sequence[str]) -> List[torch.Tensor]:
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
        )

        encoded: List[torch.Tensor] = []
        for i in range(reps.shape[0]):
            valid = mask[i]
            token_reps = reps[i][valid]
            if token_reps.shape[0] == 0:
                # Keep downstream code shape-safe for pathological inputs.
                token_reps = reps[i, 1:2]
            encoded.append(token_reps)
        return encoded

    def _forward_esm_efficient(self, sequences: Sequence[str]) -> List[torch.Tensor]:
        if self._esme_tokenize is None or self._esme_alphabet is None:
            raise RuntimeError("ESM-efficient tokenization helpers are not initialized.")

        seqs = [_sanitize_aa_sequence(s) for s in sequences]
        toks = self._esme_tokenize(seqs, alphabet=self._esme_alphabet)
        device = next(self.model.parameters()).device
        toks = toks.to(device=device, non_blocking=True)

        reps = self.model.forward_representation(toks, pad_output=True)  # [B, T, D]

        mask = (
            (toks != self._esme_alphabet.padding_idx)
            & (toks != self._esme_alphabet.cls_idx)
            & (toks != self._esme_alphabet.eos_idx)
        )

        encoded: List[torch.Tensor] = []
        for i in range(reps.shape[0]):
            valid = mask[i]
            token_reps = reps[i][valid]
            if token_reps.shape[0] == 0:
                token_reps = reps[i, 1:2]
            encoded.append(token_reps)
        return encoded

    def forward(self, sequences: Sequence[str]) -> List[torch.Tensor]:
        if not sequences:
            return []
        if self.backend == "esm_efficient":
            return self._forward_esm_efficient(sequences)
        return self._forward_fair_esm(sequences)
