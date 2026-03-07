from __future__ import annotations

import glob
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def _normalize_backend_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    key = str(name).strip().lower()
    if not key or key in {"none", "null"}:
        return None
    if key not in {"fair_esm", "esm_efficient"}:
        raise ValueError(
            f"Unsupported ESM1b backend={name!r}. "
            "Expected one of {'fair_esm', 'esm_efficient'}."
        )
    return key


def _resolve_torch_hub_checkpoints_dir(torch_hub_dir: Optional[str]) -> Path:
    if torch_hub_dir:
        root = Path(torch_hub_dir).expanduser()
        if root.name == "checkpoints":
            return root
        if root.name == "hub":
            return root / "checkpoints"
        if (root / "checkpoints").exists():
            return root / "checkpoints"
        return root

    torch_home = os.environ.get("TORCH_HOME", None)
    if torch_home:
        return Path(torch_home).expanduser() / "hub" / "checkpoints"

    return Path(torch.hub.get_dir()) / "checkpoints"


def _resolve_hf_hub_root(hf_cache_dir: Optional[str]) -> Path:
    if hf_cache_dir:
        root = Path(hf_cache_dir).expanduser()
        if root.name == "hub":
            return root
        return root / "hub"

    hf_home = os.environ.get("HF_HOME", None)
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", None)
    if hub_cache:
        return Path(hub_cache).expanduser()

    return Path.home() / ".cache" / "huggingface" / "hub"


def _resolve_fair_esm_checkpoint_path(
    model_name: str,
    torch_hub_dir: Optional[str],
) -> Optional[Path]:
    raw = Path(str(model_name).strip()).expanduser()
    if raw.is_file():
        return raw

    stem = raw.stem if raw.suffix else raw.name
    if not stem:
        return None

    checkpoints_dir = _resolve_torch_hub_checkpoints_dir(torch_hub_dir)
    candidate = checkpoints_dir / f"{stem}.pt"
    regression = checkpoints_dir / f"{stem}-contact-regression.pt"
    if candidate.is_file() and (regression.is_file() or ("esm1v" in stem or "esm_if" in stem)):
        return candidate
    return None


def _esm_efficient_filename_for_name(model_name: str) -> str:
    raw = Path(str(model_name).strip()).expanduser()
    if raw.is_file():
        if raw.suffix.lower() != ".safetensors":
            raise ValueError(
                f"ESM-efficient backend requires a .safetensors checkpoint, got: {raw}"
            )
        return raw.name

    name = _map_esm_efficient_model_name(model_name)
    lower = name.lower()
    if lower == "esm1b":
        return "esm1b.safetensors"
    return f"{name}.safetensors"


def _resolve_esm_efficient_checkpoint_path(
    model_name: str,
    hf_cache_dir: Optional[str],
) -> Optional[Path]:
    raw = Path(str(model_name).strip()).expanduser()
    if raw.is_file():
        if raw.suffix.lower() != ".safetensors":
            raise ValueError(
                f"ESM-efficient backend requires a .safetensors checkpoint, got: {raw}"
            )
        return raw

    filename = _esm_efficient_filename_for_name(model_name)
    hub_root = _resolve_hf_hub_root(hf_cache_dir)
    pattern = hub_root / "models--mhcelik--esm-efficient" / "snapshots" / "*" / filename
    matches = sorted(glob.glob(str(pattern)))
    if matches:
        return Path(matches[-1])
    return None


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
        fallback_backend: Optional[str] = None,
        allow_download: bool = True,
        hf_cache_dir: Optional[str] = None,
        torch_hub_dir: Optional[str] = None,
        esm_efficient_use_pretrained: bool = True,
        esm_efficient_compute_dtype: str = "bfloat16",
    ):
        super().__init__()

        backend_key = _normalize_backend_name(backend)
        if backend_key is None:
            raise ValueError("ESM1b backend must not be empty.")
        fallback_backend_key = _normalize_backend_name(fallback_backend)
        self.requested_backend = backend_key
        self.backend = backend_key

        self._init_model_with_fallback(
            backend=backend_key,
            fallback_backend=fallback_backend_key,
            model_name=model_name,
            truncation_seq_length=truncation_seq_length,
            repr_layer=repr_layer,
            allow_download=bool(allow_download),
            hf_cache_dir=hf_cache_dir,
            torch_hub_dir=torch_hub_dir,
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

    def _init_model_with_fallback(
        self,
        backend: str,
        fallback_backend: Optional[str],
        model_name: str,
        truncation_seq_length: int,
        repr_layer: int,
        allow_download: bool,
        hf_cache_dir: Optional[str],
        torch_hub_dir: Optional[str],
        use_pretrained: bool,
        compute_dtype: str,
    ) -> None:
        errors: List[Tuple[str, Exception]] = []
        candidates = [backend]
        if fallback_backend is not None and fallback_backend != backend:
            candidates.append(fallback_backend)

        for candidate_backend in candidates:
            t0 = time.time()
            try:
                logger.info(
                    "Initializing ESM1b context encoder backend=%s model_name=%s allow_download=%s",
                    candidate_backend,
                    model_name,
                    allow_download,
                )
                self._init_backend(
                    backend=candidate_backend,
                    model_name=model_name,
                    truncation_seq_length=truncation_seq_length,
                    repr_layer=repr_layer,
                    allow_download=allow_download,
                    hf_cache_dir=hf_cache_dir,
                    torch_hub_dir=torch_hub_dir,
                    use_pretrained=use_pretrained,
                    compute_dtype=compute_dtype,
                )
                self.backend = candidate_backend
                logger.info(
                    "Initialized ESM1b context encoder backend=%s in %.2fs",
                    candidate_backend,
                    time.time() - t0,
                )
                return
            except Exception as exc:
                errors.append((candidate_backend, exc))
                logger.warning(
                    "Failed to initialize ESM1b context encoder backend=%s: %s",
                    candidate_backend,
                    exc,
                )

        details = "; ".join(f"{name}: {type(exc).__name__}: {exc}" for name, exc in errors)
        raise RuntimeError(f"Unable to initialize any ESM1b backend. Tried [{details}]") from errors[-1][1]

    def _init_backend(
        self,
        backend: str,
        model_name: str,
        truncation_seq_length: int,
        repr_layer: int,
        allow_download: bool,
        hf_cache_dir: Optional[str],
        torch_hub_dir: Optional[str],
        use_pretrained: bool,
        compute_dtype: str,
    ) -> None:
        if backend == "fair_esm":
            self._init_fair_esm_model(
                model_name=model_name,
                truncation_seq_length=truncation_seq_length,
                repr_layer=repr_layer,
                allow_download=allow_download,
                torch_hub_dir=torch_hub_dir,
            )
            return

        self._init_esm_efficient_model(
            model_name=model_name,
            repr_layer=repr_layer,
            use_pretrained=use_pretrained,
            compute_dtype=compute_dtype,
            allow_download=allow_download,
            hf_cache_dir=hf_cache_dir,
        )

    def _init_fair_esm_model(
        self,
        model_name: str,
        truncation_seq_length: int,
        repr_layer: int,
        allow_download: bool,
        torch_hub_dir: Optional[str],
    ) -> None:
        try:
            from esm import pretrained
        except Exception as exc:
            raise RuntimeError(
                "Failed to import `esm`. Install fair-esm in the active environment."
            ) from exc

        local_path = _resolve_fair_esm_checkpoint_path(model_name, torch_hub_dir=torch_hub_dir)
        if local_path is not None:
            model, alphabet = pretrained.load_model_and_alphabet_local(str(local_path))
        elif not allow_download:
            raise FileNotFoundError(
                f"Could not resolve local fair-ESM checkpoint for model_name={model_name!r} "
                f"under checkpoints dir={_resolve_torch_hub_checkpoints_dir(torch_hub_dir)}"
            )
        else:
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
        allow_download: bool,
        hf_cache_dir: Optional[str],
    ) -> None:
        if int(repr_layer) != 33:
            raise ValueError(
                "ESM-efficient backend currently returns final-layer ESM1b representations only; "
                "set rawseq_esm1b.esm1b_repr_layer=33."
            )
        try:
            from esme import ESM1b
            from esme.alphabet import Alphabet, tokenize
        except Exception as exc:
            raise RuntimeError(
                "Failed to import `esme` (esm-efficient). Install esm-efficient in the active environment."
            ) from exc

        resolved_name = _map_esm_efficient_model_name(model_name)
        resolved_path = _resolve_esm_efficient_checkpoint_path(resolved_name, hf_cache_dir=hf_cache_dir)
        if use_pretrained:
            if resolved_path is not None:
                model = ESM1b.from_pretrained(str(resolved_path), device="cpu")
            elif not allow_download:
                raise FileNotFoundError(
                    f"Could not resolve local esm-efficient checkpoint for model_name={model_name!r} "
                    f"under cache dir={_resolve_hf_hub_root(hf_cache_dir)}"
                )
            else:
                model = ESM1b.from_pretrained(resolved_name, device="cpu")
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
