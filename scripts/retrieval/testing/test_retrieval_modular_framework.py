#!/usr/bin/env python3
"""Tests for modular retrieval framework components."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Allow running directly from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openfold.model.retrieval import (
    RetrievalController,
    RetrievalInjectionPlan,
    available_fusions,
    build_fusion,
)
from openfold.model.retrieval.interfaces import FusionStrategy
from openfold.model.retrieval.pipeline_legacy import LegacyQueryPipeline
from scripts.retrieval import train_retrieval_modular as train_mod
from scripts.retrieval.train_retrieval_modular import (
    _build_checkpoint_callbacks,
    _build_trainer_kwargs,
    _load_config,
    _resolve_trainer_resume_ckpt_path,
    _validate_model_cfg,
)

try:
    from openfold.model.retrieval.lightning_module import RetrievalAugmentedLightningModule

    _HAS_LIGHTNING_MODEL_DEPS = True
except Exception:
    RetrievalAugmentedLightningModule = None  # type: ignore
    _HAS_LIGHTNING_MODEL_DEPS = False


class FakeRetriever:
    def __init__(self, dim: int):
        self.dim = int(dim)

    def search(self, query_vec: np.ndarray, top_k: int):
        query_vec = np.asarray(query_vec, dtype=np.float32)
        scores = np.linspace(1.0, 0.2, int(top_k), dtype=np.float32)
        indices = np.arange(int(top_k), dtype=np.int64)
        scales = np.linspace(0.5, 1.5, int(top_k), dtype=np.float32).reshape(-1, 1)
        vectors = np.tile(query_vec.reshape(1, -1), (int(top_k), 1)) * scales
        return scores, indices, vectors


class FakeBackbone(nn.Module):
    def __init__(self, _config):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.input_embedder = nn.Linear(1, 1)
        self.recycling_embedder = nn.Linear(1, 1)
        self.template_embedder = nn.Linear(1, 1)
        self.extra_msa_embedder = nn.Linear(1, 1)
        self.extra_msa_stack = nn.Linear(1, 1)
        self.evoformer = nn.Linear(1, 1)
        self.structure_module = nn.Linear(1, 1)
        self.aux_heads = nn.Linear(1, 1)

    def forward(self, batch):
        x = batch["seq_embedding"][..., 0]
        smoke = (x * self.scale).pow(2).mean()
        return {
            "smoke_loss": smoke,
            "final_atom_positions": torch.zeros(x.shape[0], x.shape[1], 37, 3, device=x.device),
        }


class FakeLoss(nn.Module):
    def __init__(self, _cfg):
        super().__init__()

    def forward(self, outputs, batch, _return_breakdown=False):
        del batch
        loss = outputs["smoke_loss"]
        breakdown = {"loss": loss.detach()}
        if _return_breakdown:
            return loss, breakdown
        return loss


class FakeQueryEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(1, out_dim)

    def forward(self, sequences):
        x = torch.ones(len(sequences), 1, device=self.proj.weight.device)
        return self.proj(x)


class FakeContextEncoder(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.proj = nn.Linear(1, emb_dim)

    def forward(self, sequences):
        outs = []
        for seq in sequences:
            n = max(1, len(seq))
            x = torch.ones(n, 1, device=self.proj.weight.device)
            outs.append(self.proj(x))
        return outs

    def encode_with_ids(self, seq_pairs):
        if not seq_pairs:
            return {}
        seqs = [seq for _, seq in seq_pairs]
        embs = self.forward(seqs)
        return {sid: emb for (sid, _), emb in zip(seq_pairs, embs)}


class FakeRowLookup:
    def get(self, row_idx: int):
        return f"id_{int(row_idx)}"

    def close(self):
        return None


class FakeSequenceStore:
    def get(self, seq_id: str):
        del seq_id
        return "ACDEFGHIKLMN"

    def close(self):
        return None


class RecordingFusion(FusionStrategy):
    def __init__(self):
        super().__init__()
        self.last_query_tokens = None
        self.last_retrieved_tokens = None
        self.last_retrieved_masks = None

    @property
    def name(self) -> str:
        return "recording"

    def forward(
        self,
        query_tokens: torch.Tensor,
        retrieved_tokens: torch.Tensor,
        retrieved_scores: torch.Tensor,
        retrieved_masks: torch.Tensor | None = None,
        context: dict | None = None,
    ) -> torch.Tensor:
        del retrieved_scores, context
        self.last_query_tokens = query_tokens.detach().clone()
        self.last_retrieved_tokens = retrieved_tokens.detach().clone()
        self.last_retrieved_masks = None if retrieved_masks is None else retrieved_masks.detach().clone()
        return query_tokens


class ScoreDependentFusion(FusionStrategy):
    @property
    def name(self) -> str:
        return "score_dependent"

    def forward(
        self,
        query_tokens: torch.Tensor,
        retrieved_tokens: torch.Tensor,
        retrieved_scores: torch.Tensor,
        retrieved_masks: torch.Tensor | None = None,
        context: dict | None = None,
    ) -> torch.Tensor:
        del retrieved_tokens, retrieved_masks, context
        if retrieved_scores.numel() == 0:
            return query_tokens
        weights = torch.arange(
            1,
            int(retrieved_scores.shape[0]) + 1,
            device=query_tokens.device,
            dtype=query_tokens.dtype,
        )
        bias = (retrieved_scores.to(dtype=query_tokens.dtype) * weights).sum()
        return query_tokens + bias * torch.ones_like(query_tokens)


class TinyDataset(Dataset):
    def __init__(self, n_samples: int = 4, n_res: int = 6, emb_dim: int = 32, n_recycles: int = 2):
        self.items = []
        for i in range(n_samples):
            self.items.append(
                {
                    "seq_embedding": torch.randn(n_res, emb_dim, n_recycles),
                    "seq_mask": torch.ones(n_res, n_recycles),
                    "raw_sequence": "ACDEFGHIKLMNPQRS"[: (8 + (i % 8))],
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _collate_mixed(samples):
    out = {}
    for k in samples[0].keys():
        v0 = samples[0][k]
        if torch.is_tensor(v0):
            out[k] = torch.stack([s[k] for s in samples], dim=0)
        else:
            out[k] = [s[k] for s in samples]
    return out


class TinyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 2):
        super().__init__()
        self.batch_size = batch_size
        self.train_ds = TinyDataset(n_samples=4)
        self.val_ds = TinyDataset(n_samples=2)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=False, collate_fn=_collate_mixed)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, collate_fn=_collate_mixed)


def test_fusion_registry_and_unknown_name():
    names = set(available_fusions())
    if "simple_cross_attn" not in names:
        raise AssertionError("simple_cross_attn missing from fusion registry")
    if "rag_esm_inspired" not in names:
        raise AssertionError("rag_esm_inspired missing from fusion registry")
    if "rag_esm_port" not in names:
        raise AssertionError("rag_esm_port missing from fusion registry")

    try:
        _ = build_fusion("does_not_exist", emb_dim=32)
        raise AssertionError("Unknown fusion name should raise")
    except KeyError:
        pass


def test_fusions_forward_and_gradients():
    torch.manual_seed(0)
    query = torch.randn(7, 32, requires_grad=True)
    retrieved = torch.randn(3, 5, 32)
    scores = torch.softmax(torch.randn(3), dim=0)
    masks = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)

    for name in ("simple_cross_attn", "rag_esm_inspired", "rag_esm_port"):
        kwargs = {"emb_dim": 32, "num_heads": 8, "dropout": 0.0}
        if name == "rag_esm_port":
            kwargs.update({"num_blocks": 2, "skip_cross_ratio": 0.0})
        fusion = build_fusion(name, **kwargs)
        out = fusion(query, retrieved, scores, retrieved_masks=masks)
        if out.shape != query.shape:
            raise AssertionError(f"{name}: output shape mismatch")
        if not torch.isfinite(out).all():
            raise AssertionError(f"{name}: non-finite output")

        loss = out.pow(2).mean()
        loss.backward(retain_graph=True)
        has_grad = any((p.grad is not None and torch.isfinite(p.grad).all()) for p in fusion.parameters())
        if not has_grad:
            raise AssertionError(f"{name}: no finite gradients on parameters")


def test_rag_fusions_support_attention_backend_override():
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        print("[SKIP] sdpa attention backend override test: torch SDPA unavailable")
        return

    torch.manual_seed(0)
    query = torch.randn(7, 32, requires_grad=True)
    retrieved = torch.randn(3, 5, 32)
    scores = torch.softmax(torch.randn(3), dim=0)
    masks = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)

    for name in ("rag_esm_inspired", "rag_esm_port"):
        kwargs = {
            "emb_dim": 32,
            "num_heads": 8,
            "dropout": 0.0,
            "attention_backend": "sdpa",
        }
        if name == "rag_esm_port":
            kwargs.update({"num_blocks": 2, "skip_cross_ratio": 0.0})
        fusion = build_fusion(name, **kwargs)
        out = fusion(query, retrieved, scores, retrieved_masks=masks)
        if out.shape != query.shape:
            raise AssertionError(f"{name}: output shape mismatch for sdpa backend")
        if not torch.isfinite(out).all():
            raise AssertionError(f"{name}: non-finite output for sdpa backend")

        loss = out.pow(2).mean()
        loss.backward(retain_graph=True)
        has_grad = any((p.grad is not None and torch.isfinite(p.grad).all()) for p in fusion.parameters())
        if not has_grad:
            raise AssertionError(f"{name}: no finite gradients for sdpa backend")


def test_multistage_injection_shapes_and_order():
    plan = RetrievalInjectionPlan(
        seq_embedding_dim=32,
        c_m=16,
        c_s=24,
        stages=("pre_structure", "input", "pre_evoformer"),
    )

    if plan.ordered_stages != ("input", "pre_evoformer", "pre_structure"):
        raise AssertionError("Injection stage ordering is not deterministic")

    batch = {
        "seq_embedding": torch.randn(2, 6, 32, 3),
    }
    fused = torch.randn(2, 6, 32)
    out = plan.apply(batch, fused)

    if out["seq_embedding"].shape != (2, 6, 32, 3):
        raise AssertionError("input injection shape mismatch")
    if out["retrieval_pre_evoformer"].shape != (2, 6, 16, 3):
        raise AssertionError("pre_evoformer injection shape mismatch")
    if out["retrieval_pre_structure"].shape != (2, 6, 24, 3):
        raise AssertionError("pre_structure injection shape mismatch")


def test_controller_outputs_and_source_mixing_gradients():
    torch.manual_seed(0)

    query_pipeline = LegacyQueryPipeline(seq_embedding_dim=32, seq_index_dim=32, struct_index_dim=16)
    controller = RetrievalController(
        openfold=FakeBackbone(None),
        query_pipeline=query_pipeline,
        seq_fusion=build_fusion("simple_cross_attn", emb_dim=32, num_heads=8, dropout=0.0),
        struct_fusion=build_fusion("simple_cross_attn", emb_dim=32, num_heads=8, dropout=0.0),
        injection_plan=RetrievalInjectionPlan(seq_embedding_dim=32, c_m=48, c_s=64, stages=("input",)),
        top_k=3,
        retrieval_ablation="both",
        seq_retriever=FakeRetriever(dim=32),
        struct_retriever=FakeRetriever(dim=16),
        seq_db_proj=nn.Linear(32, 32),
        struct_db_proj=nn.Linear(16, 32),
    )

    batch = {
        "seq_embedding": torch.randn(2, 6, 32, 2),
        "seq_mask": torch.ones(2, 6, 2),
    }
    outputs = controller(batch)

    if outputs["seq_retrieval_scores"].shape != (2, 3):
        raise AssertionError("seq retrieval score shape mismatch")
    if outputs["struct_retrieval_scores"].shape != (2, 3):
        raise AssertionError("struct retrieval score shape mismatch")

    loss = outputs["smoke_loss"]
    loss.backward()

    if controller.source_mix_logits.grad is None:
        raise AssertionError("source mixing logits did not receive gradients")
    if query_pipeline.seq_query_proj.weight.grad is None:
        raise AssertionError("legacy seq query projection did not receive gradients")
    if controller.seq_db_proj.weight.grad is None:
        raise AssertionError("seq db projection did not receive gradients")


def test_embed_project_pipeline_query_encoder_gradients_and_fit_smoke():
    if not _HAS_LIGHTNING_MODEL_DEPS:
        print("[SKIP] lightning smoke test: missing optional OpenFold training deps")
        return

    torch.manual_seed(0)
    pl.seed_everything(0, workers=True)

    model = RetrievalAugmentedLightningModule(
        config_preset="seqemb_initial_training",
        seq_embedding_dim=32,
        top_k=3,
        lr=1e-2,
        struct_index_path=None,
        struct_index_dim=16,
        seq_index_path=None,
        seq_index_dim=32,
        retrieval_ablation="both",
        retrieval_pipeline="embed_project",
        fusion_name="rag_esm_inspired",
        fusion_params={"num_heads": 8, "dropout": 0.0, "mlp_hidden_mult": 2},
        retrieval_injection_stages=("input", "pre_evoformer", "pre_structure"),
        freeze_backbone=False,
        backbone_factory=FakeBackbone,
        loss_factory=FakeLoss,
        seq_query_encoder_override=FakeQueryEncoder(32),
        struct_query_encoder_override=FakeQueryEncoder(16),
    )
    model.controller.seq_retriever = FakeRetriever(dim=32)
    model.controller.struct_retriever = FakeRetriever(dim=16)

    batch = {
        "seq_embedding": torch.randn(2, 6, 32, 2, requires_grad=True),
        "seq_mask": torch.ones(2, 6, 2),
        "raw_sequence": ["ACDEFG", "LMNPQR"],
    }
    optimizer = model.configure_optimizers()
    optimizer.zero_grad(set_to_none=True)
    loss = model.training_step(batch, 0)
    loss.backward()

    seq_encoder = model.controller.query_pipeline.seq_encoder
    struct_encoder = model.controller.query_pipeline.struct_encoder
    if seq_encoder is None or struct_encoder is None:
        raise AssertionError("embed_project query encoders are not configured")
    if seq_encoder.proj.weight.grad is None:
        raise AssertionError("sequence query encoder did not receive gradients")
    if struct_encoder.proj.weight.grad is None:
        raise AssertionError("structure query encoder did not receive gradients")

    optimizer.step()

    dm = TinyDataModule(batch_size=2)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)


def test_rawseq_ragstyle_pipeline_context_encoder_gradients():
    if not _HAS_LIGHTNING_MODEL_DEPS:
        print("[SKIP] rawseq ragstyle smoke test: missing optional OpenFold training deps")
        return

    torch.manual_seed(0)
    model = RetrievalAugmentedLightningModule(
        config_preset="seqemb_initial_training",
        seq_embedding_dim=32,
        top_k=3,
        lr=1e-2,
        struct_index_path=None,
        struct_index_dim=16,
        seq_index_path="dummy.seq.index",
        seq_index_dim=32,
        retrieval_ablation="seq_only",
        retrieval_pipeline="rawseq_esm1b_ragstyle",
        fusion_name="rag_esm_port",
        fusion_params={
            "num_heads": 8,
            "dropout": 0.0,
            "mlp_hidden_mult": 2,
            "num_blocks": 2,
            "layers_with_cross_attention": "all",
            "skip_cross_ratio": 0.0,
        },
        retrieval_injection_stages=("input",),
        rawseq_seq_index_ids_path="unused_ids.txt",
        rawseq_seq_db_fasta_path="unused.fasta",
        rawseq_seq_db_fasta_index_db="unused.idx.sqlite",
        backbone_factory=FakeBackbone,
        loss_factory=FakeLoss,
        freeze_backbone=False,
        seq_query_encoder_override=FakeQueryEncoder(32),
        rawseq_seq_row_lookup_override=FakeRowLookup(),
        rawseq_seq_sequence_store_override=FakeSequenceStore(),
        rawseq_seq_context_encoder_override=FakeContextEncoder(32),
    )
    model.controller.seq_retriever = FakeRetriever(dim=32)
    model.log = lambda *args, **kwargs: None

    batch = {
        "seq_embedding": torch.randn(2, 6, 32, 2, requires_grad=True),
        "seq_mask": torch.ones(2, 6, 2),
        "raw_sequence": ["ACDEFGHIK", "LMNPQRSTV"],
    }
    optimizer = model.configure_optimizers()
    optimizer.zero_grad(set_to_none=True)
    loss = model.training_step(batch, 0)
    loss.backward()

    ctx = model.controller.seq_context_encoder
    if not isinstance(ctx, FakeContextEncoder):
        raise AssertionError("Expected fake context encoder override")
    if ctx.proj.weight.grad is None:
        raise AssertionError("rawseq ragstyle context encoder did not receive gradients")

    outputs = model(batch)
    if "retrieval_valid_hits" not in outputs:
        raise AssertionError("Missing retrieval_valid_hits metric output")
    if "retrieval_context_tokens" not in outputs:
        raise AssertionError("Missing retrieval_context_tokens metric output")
    if "retrieval_skip_cross_applied" not in outputs:
        raise AssertionError("Missing retrieval_skip_cross_applied metric output")

    optimizer.step()


def test_rawseq_ragstyle_uses_esm1b_query_tokens_for_fusion():
    if not _HAS_LIGHTNING_MODEL_DEPS:
        print("[SKIP] rawseq query-token fusion test: missing optional OpenFold training deps")
        return

    torch.manual_seed(0)
    model = RetrievalAugmentedLightningModule(
        config_preset="seqemb_initial_training",
        seq_embedding_dim=32,
        top_k=3,
        lr=1e-2,
        struct_index_path=None,
        struct_index_dim=16,
        seq_index_path="dummy.seq.index",
        seq_index_dim=32,
        retrieval_ablation="seq_only",
        retrieval_pipeline="rawseq_esm1b_ragstyle",
        fusion_name="rag_esm_port",
        fusion_params={
            "num_heads": 8,
            "dropout": 0.0,
            "mlp_hidden_mult": 2,
            "num_blocks": 2,
            "layers_with_cross_attention": "all",
            "skip_cross_ratio": 0.0,
        },
        retrieval_injection_stages=("input",),
        rawseq_seq_index_ids_path="unused_ids.txt",
        rawseq_seq_db_fasta_path="unused.fasta",
        rawseq_seq_db_fasta_index_db="unused.idx.sqlite",
        backbone_factory=FakeBackbone,
        loss_factory=FakeLoss,
        freeze_backbone=False,
        seq_query_encoder_override=FakeQueryEncoder(32),
        rawseq_seq_row_lookup_override=FakeRowLookup(),
        rawseq_seq_sequence_store_override=FakeSequenceStore(),
        rawseq_seq_context_encoder_override=FakeContextEncoder(32),
    )
    model.controller.seq_retriever = FakeRetriever(dim=32)
    model.log = lambda *args, **kwargs: None
    recording_fusion = RecordingFusion()
    model.controller.seq_fusion = recording_fusion

    batch = {
        "seq_embedding": torch.randn(1, 9, 32, 2),
        "seq_mask": torch.ones(1, 9, 2),
        "raw_sequence": ["ACDEFGHIK"],
    }
    _ = model(batch)

    actual = recording_fusion.last_query_tokens
    if actual is None:
        raise AssertionError("Fusion module did not receive query tokens")

    ctx = model.controller.seq_context_encoder
    if not isinstance(ctx, FakeContextEncoder):
        raise AssertionError("Expected fake context encoder override")

    expected = ctx.forward(batch["raw_sequence"])[0]
    if expected.shape[0] != batch["seq_embedding"].shape[1]:
        raise AssertionError("Test setup expects query token length to match sequence length")

    original = batch["seq_embedding"][0, :, :, 0]
    if torch.allclose(actual, original):
        raise AssertionError("Fusion still received original seq_embedding tokens instead of ESM1b query tokens")
    if not torch.allclose(actual, expected, atol=1e-5, rtol=1e-5):
        raise AssertionError("Fusion query tokens do not match ESM1b-encoded raw query sequence")


def test_rawseq_ragstyle_seq_query_encoder_gets_gradients():
    if not _HAS_LIGHTNING_MODEL_DEPS:
        print("[SKIP] rawseq seq-query gradient test: missing optional OpenFold training deps")
        return

    torch.manual_seed(0)
    model = RetrievalAugmentedLightningModule(
        config_preset="seqemb_initial_training",
        seq_embedding_dim=32,
        top_k=3,
        lr=1e-2,
        struct_index_path=None,
        struct_index_dim=16,
        seq_index_path="dummy.seq.index",
        seq_index_dim=32,
        retrieval_ablation="seq_only",
        retrieval_pipeline="rawseq_esm1b_ragstyle",
        fusion_name="rag_esm_port",
        fusion_params={
            "num_heads": 8,
            "dropout": 0.0,
            "mlp_hidden_mult": 2,
            "num_blocks": 2,
            "layers_with_cross_attention": "all",
            "skip_cross_ratio": 0.0,
        },
        retrieval_injection_stages=("input",),
        rawseq_seq_index_ids_path="unused_ids.txt",
        rawseq_seq_db_fasta_path="unused.fasta",
        rawseq_seq_db_fasta_index_db="unused.idx.sqlite",
        backbone_factory=FakeBackbone,
        loss_factory=FakeLoss,
        freeze_backbone=False,
        seq_query_encoder_override=FakeQueryEncoder(32),
        rawseq_seq_row_lookup_override=FakeRowLookup(),
        rawseq_seq_sequence_store_override=FakeSequenceStore(),
        rawseq_seq_context_encoder_override=FakeContextEncoder(32),
    )
    model.controller.seq_retriever = FakeRetriever(dim=32)
    model.controller.seq_fusion = ScoreDependentFusion()
    model.log = lambda *args, **kwargs: None

    batch = {
        "seq_embedding": torch.randn(2, 6, 32, 2, requires_grad=True),
        "seq_mask": torch.ones(2, 6, 2),
        "raw_sequence": ["ACDEFGHIK", "LMNPQRSTV"],
    }
    optimizer = model.configure_optimizers()
    optimizer.zero_grad(set_to_none=True)
    loss = model.training_step(batch, 0)
    loss.backward()

    seq_encoder = model.controller.query_pipeline.seq_encoder
    if not isinstance(seq_encoder, FakeQueryEncoder):
        raise AssertionError("Expected fake sequence query encoder override")
    grad = seq_encoder.proj.weight.grad
    if grad is None:
        raise AssertionError("Rawseq seq query encoder did not receive gradients")
    if not torch.isfinite(grad).all():
        raise AssertionError("Rawseq seq query encoder gradient contains non-finite values")
    if float(grad.abs().sum().item()) == 0.0:
        raise AssertionError("Rawseq seq query encoder gradient is identically zero")


def test_invalid_frozen_backbone_config_rejected():
    cfg = {
        "model": {
            "freeze_backbone": True,
            "train_openfold_all": False,
            "trainable_backbone_modules": [],
            "openfold_checkpoint": None,
        }
    }
    try:
        _validate_model_cfg(cfg, base_dir=Path("."))
        raise AssertionError("Expected invalid frozen-backbone config to raise ValueError")
    except ValueError:
        pass


def test_auto_checkpoint_resolution_and_optimizer_param_groups():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        ckpt_dir = base / "outputs" / "exp1"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "last.ckpt"
        ckpt_path.write_bytes(b"fake")

        cfg = {
            "model": {
                "freeze_backbone": True,
                "train_openfold_all": False,
                "trainable_backbone_modules": [],
                "openfold_checkpoint": "auto",
                "openfold_checkpoint_search_globs": ["outputs/**/*.ckpt"],
            }
        }
        validated = _validate_model_cfg(cfg, base_dir=base)
        resolved = validated["model"]["openfold_checkpoint"]
        if resolved is None or Path(resolved).resolve() != ckpt_path.resolve():
            raise AssertionError("Auto checkpoint resolution did not pick expected checkpoint")

    if not _HAS_LIGHTNING_MODEL_DEPS:
        print("[SKIP] optimizer param-group test: missing optional OpenFold training deps")
        return

    model = RetrievalAugmentedLightningModule(
        config_preset="seqemb_initial_training",
        seq_embedding_dim=32,
        top_k=3,
        lr=1e-3,
        controller_lr=2e-3,
        backbone_lr=5e-5,
        seq_index_path=None,
        struct_index_path=None,
        freeze_backbone=False,
        train_openfold_all=True,
        backbone_factory=FakeBackbone,
        loss_factory=FakeLoss,
    )
    optimizer = model.configure_optimizers()
    lrs = sorted({float(group["lr"]) for group in optimizer.param_groups})
    if lrs != [5e-05, 0.002]:
        raise AssertionError(f"Unexpected optimizer group learning rates: {lrs}")


def test_trainer_kwargs_and_checkpoint_callback_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        output_dir = base / "run"
        output_dir.mkdir(parents=True, exist_ok=True)
        trainer_cfg = {
            "accelerator": "gpu",
            "devices": 4,
            "num_nodes": 1,
            "strategy": "ddp",
            "precision": "32",
            "max_epochs": 1,
            "max_steps": 44069,
            "max_time": "23:45:00",
            "val_check_interval": 5000,
            "accumulate_grad_batches": 16,
            "num_sanity_val_steps": 0,
            "log_every_n_steps": 25,
            "enable_checkpointing": True,
            "checkpoint": {
                "every_n_train_steps": 500,
                "save_last": True,
                "save_top_k": -1,
                "filename": "step={step}",
                "save_on_train_epoch_end": False,
            },
        }

        callbacks = _build_checkpoint_callbacks(trainer_cfg, output_dir=output_dir, base_dir=base)
        if len(callbacks) != 1:
            raise AssertionError("Expected a single checkpoint callback")

        checkpoint_cb = callbacks[0]
        expected_dir = (output_dir / "checkpoints").resolve()
        actual_dir = Path(checkpoint_cb.dirpath).resolve()
        if actual_dir != expected_dir:
            raise AssertionError(f"Unexpected checkpoint dirpath: {actual_dir} != {expected_dir}")
        if checkpoint_cb.every_n_train_steps != 500:
            raise AssertionError("Checkpoint interval did not come from config")
        if checkpoint_cb.save_last is not True:
            raise AssertionError("Checkpoint save_last should be true")
        if checkpoint_cb.save_top_k != -1:
            raise AssertionError("Checkpoint save_top_k should be -1")
        if checkpoint_cb.filename != "step={step}":
            raise AssertionError("Checkpoint filename pattern mismatch")

        trainer_kwargs = _build_trainer_kwargs(
            trainer_cfg,
            output_dir=output_dir,
            base_dir=base,
            trainer_logger=False,
            callbacks=callbacks,
        )
        if trainer_kwargs["max_steps"] != 44069:
            raise AssertionError("trainer.max_steps not forwarded")
        if trainer_kwargs["max_time"] != "23:45:00":
            raise AssertionError("trainer.max_time not forwarded")
        if trainer_kwargs["num_nodes"] != 1:
            raise AssertionError("trainer.num_nodes not forwarded")
        if trainer_kwargs["strategy"] != "ddp":
            raise AssertionError("trainer.strategy not forwarded")
        if len(trainer_kwargs["callbacks"]) != 1:
            raise AssertionError("Expected checkpoint callback in trainer kwargs")


def test_resume_ckpt_path_resolution_prefers_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        output_dir = base / "run"
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "last.ckpt"
        ckpt_path.write_bytes(b"fake")

        resolved = _resolve_trainer_resume_ckpt_path(
            {"resume_ckpt_path": "checkpoints/last.ckpt"},
            output_dir=output_dir,
            base_dir=base,
        )
        if resolved != str(ckpt_path.resolve()):
            raise AssertionError("Resume checkpoint path was not resolved relative to output_dir")

        try:
            _resolve_trainer_resume_ckpt_path(
                {"resume_ckpt_path": "checkpoints/missing.ckpt"},
                output_dir=output_dir,
                base_dir=base,
            )
            raise AssertionError("Missing trainer.resume_ckpt_path should raise FileNotFoundError")
        except FileNotFoundError:
            pass


def test_main_forwards_resume_ckpt_and_trainer_limits():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        output_dir = base / "run"
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        openfold_ckpt = base / "seq_model_esm1b_ptm.pt"
        openfold_ckpt.write_bytes(b"fake-openfold")
        resume_ckpt = ckpt_dir / "last.ckpt"
        resume_ckpt.write_bytes(b"fake-resume")

        config_path = base / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "model:",
                    f"  openfold_checkpoint: {openfold_ckpt}",
                    "  freeze_backbone: true",
                    "  train_openfold_all: false",
                    "  trainable_backbone_modules: []",
                    "trainer:",
                    f"  output_dir: {output_dir}",
                    "  accelerator: gpu",
                    "  devices: 4",
                    "  num_nodes: 1",
                    "  strategy: ddp",
                    "  max_epochs: 1",
                    "  max_steps: 44069",
                    "  max_time: '23:45:00'",
                    "  val_check_interval: 5000",
                    "  accumulate_grad_batches: 16",
                    "  num_sanity_val_steps: 0",
                    "  log_every_n_steps: 25",
                    "  enable_checkpointing: true",
                    "  resume_ckpt_path: checkpoints/last.ckpt",
                    "  checkpoint:",
                    "    every_n_train_steps: 500",
                    "    save_last: true",
                    "    save_top_k: -1",
                    "    filename: step={step}",
                    "    save_on_train_epoch_end: false",
                ]
            )
        )

        trainer_instance = mock.Mock()
        with mock.patch.object(train_mod, "_build_model", return_value=object()), \
            mock.patch.object(train_mod, "_build_data_module", return_value=object()), \
            mock.patch.object(train_mod, "_build_trainer_logger", return_value=False), \
            mock.patch.object(train_mod.pl, "Trainer", return_value=trainer_instance) as trainer_cls, \
            mock.patch.object(sys, "argv", ["train_retrieval_modular.py", "--config", str(config_path)]):
            train_mod.main()

        trainer_kwargs = trainer_cls.call_args.kwargs
        if trainer_kwargs.get("max_steps") != 44069:
            raise AssertionError("main() did not forward trainer.max_steps")
        if trainer_kwargs.get("max_time") != "23:45:00":
            raise AssertionError("main() did not forward trainer.max_time")
        if trainer_kwargs.get("num_nodes") != 1:
            raise AssertionError("main() did not forward trainer.num_nodes")
        if trainer_kwargs.get("strategy") != "ddp":
            raise AssertionError("main() did not forward trainer.strategy")

        fit_kwargs = trainer_instance.fit.call_args.kwargs
        if fit_kwargs.get("ckpt_path") != str(resume_ckpt.resolve()):
            raise AssertionError("main() did not pass trainer.resume_ckpt_path into trainer.fit")


def test_config_loader_allows_shared_defaults_without_false_cycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        shared = base / "shared.yaml"
        parent = base / "parent.yaml"
        root = base / "root.yaml"

        shared.write_text("trainer:\n  devices: 4\n")
        parent.write_text("defaults:\n  - shared.yaml\ntrainer:\n  strategy: ddp\n")
        root.write_text("defaults:\n  - parent.yaml\n  - shared.yaml\ntrainer:\n  max_steps: 10\n")

        merged = _load_config(root)
        if merged["trainer"]["devices"] != 4:
            raise AssertionError("Shared default did not merge correctly")
        if merged["trainer"]["strategy"] != "ddp":
            raise AssertionError("Parent default did not merge correctly")
        if merged["trainer"]["max_steps"] != 10:
            raise AssertionError("Root config did not merge correctly")


def main():
    test_fusion_registry_and_unknown_name()
    test_fusions_forward_and_gradients()
    test_multistage_injection_shapes_and_order()
    test_controller_outputs_and_source_mixing_gradients()
    test_invalid_frozen_backbone_config_rejected()
    test_auto_checkpoint_resolution_and_optimizer_param_groups()
    test_trainer_kwargs_and_checkpoint_callback_config()
    test_resume_ckpt_path_resolution_prefers_output_dir()
    test_main_forwards_resume_ckpt_and_trainer_limits()
    test_config_loader_allows_shared_defaults_without_false_cycle()
    test_embed_project_pipeline_query_encoder_gradients_and_fit_smoke()
    test_rawseq_ragstyle_pipeline_context_encoder_gradients()
    test_rawseq_ragstyle_uses_esm1b_query_tokens_for_fusion()
    test_rawseq_ragstyle_seq_query_encoder_gets_gradients()
    print("[OK] modular retrieval framework tests passed")


if __name__ == "__main__":
    main()
