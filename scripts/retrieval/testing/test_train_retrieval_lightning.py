#!/usr/bin/env python3
"""Smoke tests for retrieval Lightning training module."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Allow running directly from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.retrieval.train_retrieval_lightning import RetrievalAugmentedLightningModule
from scripts.retrieval.train_retrieval_lightning import _build_trainer_logger


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
        x = batch["seq_embedding"]
        smoke = (x * self.scale).pow(2).mean()
        return {
            "smoke_loss": smoke,
            "final_atom_positions": torch.zeros(x.shape[0], x.shape[1], 37, 3, device=x.device),
        }


class FakeLoss(nn.Module):
    def __init__(self, _cfg):
        super().__init__()

    def forward(self, outputs, batch, _return_breakdown=False):
        # Depend on outputs only for a cheap gradient path.
        loss = outputs["smoke_loss"]
        breakdown = {"loss": loss.detach()}
        if _return_breakdown:
            return loss, breakdown
        return loss


class FakeRetriever:
    def __init__(self, dim: int):
        self.dim = dim

    def search(self, query_vec: np.ndarray, top_k: int):
        query_vec = query_vec.astype(np.float32)
        scores = np.linspace(1.0, 0.1, top_k, dtype=np.float32)
        indices = np.arange(top_k, dtype=np.int64)
        scales = np.linspace(1.0, 2.0, top_k, dtype=np.float32).reshape(top_k, 1)
        vectors = np.tile(query_vec.reshape(1, -1), (top_k, 1)) * scales
        return scores, indices, vectors


class FakeQueryEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(1, out_dim)

    def forward(self, sequences):
        x = torch.ones(len(sequences), 1, device=self.proj.weight.device)
        return self.proj(x)


class FakeRowLookup:
    def get(self, row_idx: int):
        return f"id_{int(row_idx)}"

    def close(self):
        return None


class FakeSequenceStore:
    def get(self, seq_id: str):
        return "ACDEFGHIK" if seq_id.startswith("id_") else None

    def close(self):
        return None


class FakeRetrievedESM1b:
    def __init__(self, emb_dim: int):
        self.emb_dim = emb_dim

    def embed(self, seq_pairs):
        out = {}
        for seq_id, seq in seq_pairs:
            n = max(1, min(len(seq), 9))
            out[seq_id] = torch.randn(n, self.emb_dim)
        return out


class TinyRetrievalDataset(Dataset):
    def __init__(self, n_samples: int = 4, n_res: int = 5, emb_dim: int = 32, n_recycles: int = 2):
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

    def __getitem__(self, idx: int):
        return self.items[idx]


def collate_mixed(samples):
    out = {}
    keys = samples[0].keys()
    for k in keys:
        v0 = samples[0][k]
        if torch.is_tensor(v0):
            out[k] = torch.stack([s[k] for s in samples], dim=0)
        else:
            out[k] = [s[k] for s in samples]
    return out


class TinyRetrievalDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 2):
        super().__init__()
        self.batch_size = batch_size
        self.train_ds = TinyRetrievalDataset(n_samples=4)
        self.val_ds = TinyRetrievalDataset(n_samples=2)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_mixed)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_mixed)


def _build_dual_source_model(retrieval_pipeline: str) -> RetrievalAugmentedLightningModule:
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
        retrieval_pipeline=retrieval_pipeline,
        backbone_factory=FakeBackbone,
        loss_factory=FakeLoss,
    )
    model.seq_retriever = FakeRetriever(dim=32)
    model.struct_retriever = FakeRetriever(dim=16)
    model.seq_query_encoder = FakeQueryEncoder(out_dim=32)
    model.struct_query_encoder = FakeQueryEncoder(out_dim=16)
    if retrieval_pipeline == "rawseq_esm1b":
        model.seq_id_lookup = FakeRowLookup()
        model.struct_id_lookup = FakeRowLookup()
        model.seq_sequence_store = FakeSequenceStore()
        model.struct_sequence_store = FakeSequenceStore()
        model.retrieved_esm1b_embedder = FakeRetrievedESM1b(emb_dim=32)
    return model


def _assert_finite_nonzero_grad(param: torch.Tensor, name: str):
    grad = param.grad
    if grad is None:
        raise AssertionError(f"{name}: missing gradient")
    if not torch.isfinite(grad).all():
        raise AssertionError(f"{name}: gradient has non-finite values")
    if grad.abs().sum().item() <= 0.0:
        raise AssertionError(f"{name}: gradient norm is zero")


def test_training_step_and_optimizer():
    torch.manual_seed(0)

    model = RetrievalAugmentedLightningModule(
        config_preset="seqemb_initial_training",
        seq_embedding_dim=32,
        top_k=3,
        lr=1e-2,
        struct_index_path=None,
        struct_index_dim=16,
        seq_index_path=None,
        seq_index_dim=32,
        retrieval_ablation="struct_only",
        backbone_factory=FakeBackbone,
        loss_factory=FakeLoss,
    )
    model.struct_retriever = FakeRetriever(dim=16)
    model.log = lambda *args, **kwargs: None

    batch = {
        "seq_embedding": torch.randn(2, 5, 32, 2, requires_grad=True),
        "seq_mask": torch.ones(2, 5, 2),
    }

    optimizer = model.configure_optimizers()
    before = model.struct_db_proj.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss = model.training_step(batch, 0)
    loss.backward()
    optimizer.step()

    after = model.struct_db_proj.weight.detach().clone()
    if torch.allclose(before, after):
        raise AssertionError("Optimizer step did not update trainable retrieval parameters")


def test_default_logger_selection():
    args = argparse.Namespace(use_wandb=False)
    logger_cfg = _build_trainer_logger(args, Path("."))
    if logger_cfg is not True:
        raise AssertionError("Default logger configuration should remain Lightning default (True)")


def test_structure_module_unfreeze_only():
    model = RetrievalAugmentedLightningModule(
        config_preset="seqemb_initial_training",
        seq_embedding_dim=32,
        top_k=3,
        lr=1e-2,
        struct_index_path=None,
        struct_index_dim=16,
        seq_index_path=None,
        seq_index_dim=32,
        retrieval_ablation="struct_only",
        train_structure_module=True,
        backbone_factory=FakeBackbone,
        loss_factory=FakeLoss,
    )

    backbone_named = dict(model.openfold.named_parameters())
    if not backbone_named:
        raise AssertionError("Fake backbone unexpectedly has no parameters")

    for name, p in backbone_named.items():
        if name.startswith("structure_module."):
            if not p.requires_grad:
                raise AssertionError("structure_module params should be trainable")
        else:
            if p.requires_grad:
                raise AssertionError(f"non-structure backbone param should stay frozen: {name}")


def test_selective_openfold_module_unfreeze():
    model = RetrievalAugmentedLightningModule(
        config_preset="seqemb_initial_training",
        seq_embedding_dim=32,
        top_k=3,
        lr=1e-2,
        struct_index_path=None,
        struct_index_dim=16,
        seq_index_path=None,
        seq_index_dim=32,
        retrieval_ablation="struct_only",
        train_input_embedder=True,
        train_evoformer=True,
        backbone_factory=FakeBackbone,
        loss_factory=FakeLoss,
    )
    trainable_prefixes = ("input_embedder.", "evoformer.")
    backbone_named = dict(model.openfold.named_parameters())
    for name, p in backbone_named.items():
        should_train = name.startswith(trainable_prefixes)
        if p.requires_grad != should_train:
            raise AssertionError(
                f"unexpected requires_grad for {name}: got={p.requires_grad} expected={should_train}"
            )


def test_openfold_all_unfreeze():
    model = RetrievalAugmentedLightningModule(
        config_preset="seqemb_initial_training",
        seq_embedding_dim=32,
        top_k=3,
        lr=1e-2,
        struct_index_path=None,
        struct_index_dim=16,
        seq_index_path=None,
        seq_index_dim=32,
        retrieval_ablation="struct_only",
        train_openfold_all=True,
        backbone_factory=FakeBackbone,
        loss_factory=FakeLoss,
    )
    backbone_named = dict(model.openfold.named_parameters())
    for name, p in backbone_named.items():
        if not p.requires_grad:
            raise AssertionError(f"all OpenFold params should be trainable under train_openfold_all: {name}")


def test_embed_project_pipeline_updates_query_encoder():
    torch.manual_seed(0)

    model = RetrievalAugmentedLightningModule(
        config_preset="seqemb_initial_training",
        seq_embedding_dim=32,
        top_k=3,
        lr=1e-2,
        struct_index_path=None,
        struct_index_dim=16,
        seq_index_path=None,
        seq_index_dim=32,
        retrieval_ablation="struct_only",
        retrieval_pipeline="embed_project",
        backbone_factory=FakeBackbone,
        loss_factory=FakeLoss,
    )
    model.struct_retriever = FakeRetriever(dim=16)
    model.struct_query_encoder = FakeQueryEncoder(out_dim=16)
    model.log = lambda *args, **kwargs: None

    batch = {
        "seq_embedding": torch.randn(2, 5, 32, 2, requires_grad=True),
        "seq_mask": torch.ones(2, 5, 2),
        "raw_sequence": ["ACDEFG", "LMNPQR"],
    }

    optimizer = model.configure_optimizers()
    optimizer.zero_grad(set_to_none=True)
    loss = model.training_step(batch, 0)
    loss.backward()
    grad = model.struct_query_encoder.proj.weight.grad
    if grad is None or not torch.isfinite(grad).all():
        raise AssertionError("embed_project path did not produce finite query-encoder gradients")
    optimizer.step()


def test_rawseq_pipeline_runs_with_fake_sequence_backends():
    torch.manual_seed(0)

    model = RetrievalAugmentedLightningModule(
        config_preset="seqemb_initial_training",
        seq_embedding_dim=32,
        top_k=3,
        lr=1e-2,
        struct_index_path=None,
        struct_index_dim=16,
        seq_index_path=None,
        seq_index_dim=32,
        retrieval_ablation="struct_only",
        retrieval_pipeline="rawseq_esm1b",
        backbone_factory=FakeBackbone,
        loss_factory=FakeLoss,
    )
    model.struct_retriever = FakeRetriever(dim=16)
    model.struct_query_encoder = FakeQueryEncoder(out_dim=16)
    model.struct_id_lookup = FakeRowLookup()
    model.struct_sequence_store = FakeSequenceStore()
    model.retrieved_esm1b_embedder = FakeRetrievedESM1b(emb_dim=32)
    model.log = lambda *args, **kwargs: None

    batch = {
        "seq_embedding": torch.randn(2, 5, 32, 2, requires_grad=True),
        "seq_mask": torch.ones(2, 5, 2),
        "raw_sequence": ["ACDEFG", "LMNPQR"],
    }

    optimizer = model.configure_optimizers()
    optimizer.zero_grad(set_to_none=True)
    loss = model.training_step(batch, 0)
    loss.backward()
    optimizer.step()


def test_gradients_propagate_to_both_query_retrievers_embed_project():
    torch.manual_seed(0)
    model = _build_dual_source_model(retrieval_pipeline="embed_project")
    model.log = lambda *args, **kwargs: None

    batch = {
        "seq_embedding": torch.randn(2, 5, 32, 2, requires_grad=True),
        "seq_mask": torch.ones(2, 5, 2),
        "raw_sequence": ["ACDEFG", "LMNPQR"],
    }

    optimizer = model.configure_optimizers()
    optimizer.zero_grad(set_to_none=True)
    loss = model.training_step(batch, 0)
    loss.backward()
    _assert_finite_nonzero_grad(model.seq_query_encoder.proj.weight, "embed_project/seq_query_encoder")
    _assert_finite_nonzero_grad(model.struct_query_encoder.proj.weight, "embed_project/struct_query_encoder")
    optimizer.step()


def test_gradients_propagate_to_both_query_retrievers_rawseq():
    torch.manual_seed(0)
    model = _build_dual_source_model(retrieval_pipeline="rawseq_esm1b")
    model.log = lambda *args, **kwargs: None

    batch = {
        "seq_embedding": torch.randn(2, 5, 32, 2, requires_grad=True),
        "seq_mask": torch.ones(2, 5, 2),
        "raw_sequence": ["ACDEFG", "LMNPQR"],
    }

    optimizer = model.configure_optimizers()
    optimizer.zero_grad(set_to_none=True)
    loss = model.training_step(batch, 0)
    loss.backward()
    _assert_finite_nonzero_grad(model.seq_query_encoder.proj.weight, "rawseq/seq_query_encoder")
    _assert_finite_nonzero_grad(model.struct_query_encoder.proj.weight, "rawseq/struct_query_encoder")
    optimizer.step()


def test_pipeline_end_to_end_lightning_fit_smoke():
    torch.manual_seed(0)
    pl.seed_everything(0, workers=True)

    model = _build_dual_source_model(retrieval_pipeline="embed_project")
    data_module = TinyRetrievalDataModule(batch_size=2)

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
    trainer.fit(model, datamodule=data_module)


def test_pipeline_end_to_end_lightning_fit_smoke_rawseq():
    torch.manual_seed(0)
    pl.seed_everything(0, workers=True)

    model = _build_dual_source_model(retrieval_pipeline="rawseq_esm1b")
    data_module = TinyRetrievalDataModule(batch_size=2)

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
    trainer.fit(model, datamodule=data_module)


def main():
    test_training_step_and_optimizer()
    test_default_logger_selection()
    test_structure_module_unfreeze_only()
    test_selective_openfold_module_unfreeze()
    test_openfold_all_unfreeze()
    test_embed_project_pipeline_updates_query_encoder()
    test_rawseq_pipeline_runs_with_fake_sequence_backends()
    test_gradients_propagate_to_both_query_retrievers_embed_project()
    test_gradients_propagate_to_both_query_retrievers_rawseq()
    test_pipeline_end_to_end_lightning_fit_smoke()
    test_pipeline_end_to_end_lightning_fit_smoke_rawseq()
    print("[OK] retrieval lightning module tests passed")


if __name__ == "__main__":
    main()
