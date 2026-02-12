"""
Training script for Retrieval-Augmented Folding on top of frozen OpenFold SoloSeq.

Mirrors the structure of ``train_openfold.py`` but:
  * Freezes the AlphaFold backbone (loaded from a pretrained SoloSeq checkpoint).
  * Trains only the ``EmbeddingRetriever`` + ``CrossAttentionFusion`` modules.
  * Loads a database of precomputed ESM-1b embeddings at startup.
  * On each step, retrieves top-K database embeddings, fuses them with the
    query, and feeds the result through the frozen backbone.

Usage example:
    python train_retrieval_fusion.py \\
        train_data_dir \\
        train_alignment_dir \\
        template_mmcif_dir \\
        output_dir \\
        2021-10-10 \\
        --embedding_db_dir /path/to/esm1b_embeddings_db/ \\
        --openfold_checkpoint /path/to/soloseq_checkpoint.pt \\
        --config_preset seqemb_initial_training \\
        --top_k 16 \\
        --precision bf16 \\
        --gpus 1
"""

import argparse
import json
import logging
import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

from openfold.config import model_config
from openfold.data.data_modules import (
    OpenFoldDataModule,
)
from openfold.model.model import AlphaFold
from openfold.model.retrieval_augmented_folding import RetrievalAugmentedFolding
from openfold.utils.callbacks import EarlyStoppingVerbose
from openfold.utils.import_weights import import_openfold_weights_
from openfold.utils.loss import AlphaFoldLoss
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold.utils.tensor_utils import tensor_tree_map

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding database
# ---------------------------------------------------------------------------

def load_embedding_database(
    db_dir: str,
    max_entries: int = -1,
    max_seq_len: int = 512,
    device: str = "cpu",
):
    """Load a directory of precomputed ESM-1b ``.pt`` files into a padded tensor.

    Each ``.pt`` file is expected to be a dict with a ``"representations"`` key
    containing a dict mapping layer index → tensor.  We use layer 33 (the
    final layer of ESM-1b), matching OpenFold's ``precompute_embeddings.py``.

    Args:
        db_dir:       Directory containing ``.pt`` files.
        max_entries:  Maximum number of entries to load (-1 = all).
        max_seq_len:  Pad/truncate all embeddings to this length.
        device:       Device to store the database on.

    Returns:
        db_embs:  [M, max_seq_len, 1280] float tensor.
        db_masks: [M, max_seq_len] bool tensor (1 = valid residue).
        db_names: list of str, filenames without extension.
    """
    pt_files = sorted(
        [f for f in os.listdir(db_dir) if f.endswith(".pt")]
    )
    if max_entries > 0:
        pt_files = pt_files[:max_entries]

    embs, masks, names = [], [], []
    for fname in pt_files:
        data = torch.load(os.path.join(db_dir, fname), map_location="cpu")
        rep = data["representations"][33]  # [L, 1280]
        L = rep.shape[0]
        if L > max_seq_len:
            rep = rep[:max_seq_len]
            L = max_seq_len
        # Pad
        pad_len = max_seq_len - L
        padded = torch.nn.functional.pad(rep, (0, 0, 0, pad_len))
        mask = torch.zeros(max_seq_len)
        mask[:L] = 1.0

        embs.append(padded)
        masks.append(mask)
        names.append(os.path.splitext(fname)[0])

    db_embs = torch.stack(embs, dim=0).to(device)    # [M, max_seq_len, 1280]
    db_masks = torch.stack(masks, dim=0).to(device)   # [M, max_seq_len]
    logger.info(f"Loaded embedding database: {db_embs.shape[0]} entries, "
                f"max_seq_len={max_seq_len}")
    return db_embs, db_masks, names


# ---------------------------------------------------------------------------
# Lightning wrapper
# ---------------------------------------------------------------------------

class RetrievalFusionWrapper(pl.LightningModule):
    """PyTorch Lightning module for training retrieval-augmented folding.

    Wraps ``RetrievalAugmentedFolding`` which contains:
      * A frozen ``AlphaFold`` backbone (SoloSeq mode).
      * Trainable ``EmbeddingRetriever`` + ``CrossAttentionFusion``.

    The database of embeddings is stored as a buffer (not a parameter) so
    that it lives on the correct device but isn't optimised.

    Args:
        config:     Full OpenFold config (should use a ``seq*`` preset).
        db_embs:    [M, N_d, D] embedding database tensor.
        db_masks:   [M, N_d]    residue masks for database entries.
        top_k:      Number of entries to retrieve per query.
        lr:         Learning rate for the trainable modules.
    """

    def __init__(
        self,
        config,
        db_embs: torch.Tensor,
        db_masks: torch.Tensor,
        top_k: int = 16,
        retriever_proj: int = 128,
        fusion_heads: int = 8,
        fusion_dropout: float = 0.0,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config", "db_embs", "db_masks"])
        self.config = config
        self.lr = lr

        # Embedding database — register as buffers so .to(device) works
        self.register_buffer("db_embs", db_embs)
        self.register_buffer("db_masks", db_masks)

        # Model: frozen backbone + trainable retriever & fusion
        self.model = RetrievalAugmentedFolding(
            config=config,
            emb_dim=1280,
            retriever_proj=retriever_proj,
            top_k=top_k,
            fusion_heads=fusion_heads,
            fusion_dropout=fusion_dropout,
        )

        # Loss: reuse OpenFold's composite loss
        self.loss_fn = AlphaFoldLoss(config.loss)

    def forward(self, batch):
        return self.model(batch, self.db_embs, self.db_masks)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)

        # Remove recycling dimension (keep last)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        loss, loss_breakdown = self.loss_fn(
            outputs, batch, _return_breakdown=True,
        )

        # Log losses
        for name, val in loss_breakdown.items():
            self.log(f"train/{name}", val, prog_bar=(name == "loss"),
                     on_step=True, on_epoch=False, logger=True)

        # Log retrieval diagnostics
        if "retrieval_scores" in outputs:
            self.log("train/retrieval_entropy",
                     -(outputs["retrieval_scores"] *
                       outputs["retrieval_scores"].clamp(min=1e-8).log()).sum(),
                     on_step=True, on_epoch=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)
        batch["use_clamped_fape"] = 0.0

        _, loss_breakdown = self.loss_fn(
            outputs, batch, _return_breakdown=True,
        )

        for name, val in loss_breakdown.items():
            self.log(f"val/{name}", val, prog_bar=(name == "loss"),
                     on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        # Only optimise the retriever + fusion parameters
        optimizer = torch.optim.Adam(
            self.model.trainable_parameters(),
            lr=self.lr,
            eps=1e-5,
        )
        scheduler = AlphaFoldLRScheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "AlphaFoldLRScheduler",
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    is_low_precision = args.precision in [
        "bf16-mixed", "16", "bf16", "16-true", "16-mixed", "bf16-mixed",
    ]

    # --- Config (must be a seqemb preset) ----------------------------------
    config = model_config(
        args.config_preset,
        train=True,
        low_prec=is_low_precision,
    )
    if args.experiment_config_json:
        with open(args.experiment_config_json, "r") as f:
            config.update_from_flattened_dict(json.load(f))

    if not config.globals.seqemb_mode_enabled:
        raise ValueError(
            f"config_preset={args.config_preset!r} does not enable seqemb mode. "
            "Use a 'seq*' preset (e.g. 'seqemb_initial_training')."
        )

    # --- Embedding database ------------------------------------------------
    db_embs, db_masks, _ = load_embedding_database(
        args.embedding_db_dir,
        max_entries=args.max_db_entries,
        max_seq_len=args.db_max_seq_len,
    )

    # --- Lightning module --------------------------------------------------
    model_module = RetrievalFusionWrapper(
        config=config,
        db_embs=db_embs,
        db_masks=db_masks,
        top_k=args.top_k,
        retriever_proj=args.retriever_proj_dim,
        fusion_heads=args.fusion_heads,
        fusion_dropout=args.fusion_dropout,
        lr=args.lr,
    )

    # --- Load pretrained OpenFold SoloSeq weights --------------------------
    if args.openfold_checkpoint:
        sd = torch.load(args.openfold_checkpoint, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        # Keys in the checkpoint are like "model.xxx"; our wrapper nests
        # them under "model.openfold.xxx"
        remapped = {}
        for k, v in sd.items():
            new_key = k
            # Handle "model.xxx" → "model.openfold.xxx"
            if k.startswith("model."):
                new_key = "model.openfold." + k[len("model."):]
            elif not k.startswith("model.openfold."):
                new_key = "model.openfold." + k
            remapped[new_key] = v

        missing, unexpected = model_module.load_state_dict(remapped, strict=False)
        # We expect the retriever + fusion keys to be missing
        trainable_prefixes = ("model.retriever.", "model.fusion.")
        real_missing = [k for k in missing
                        if not any(k.startswith(p) for p in trainable_prefixes)]
        if real_missing:
            logger.warning(f"Missing backbone keys: {real_missing[:10]}...")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected[:10]}...")
        logger.info("Loaded pretrained OpenFold SoloSeq checkpoint.")

    n_trainable = model_module.model.num_trainable_params()
    n_total = sum(p.numel() for p in model_module.parameters())
    logger.info(f"Trainable parameters: {n_trainable:,} / {n_total:,} total "
                f"({100 * n_trainable / n_total:.2f}%)")

    # --- Data module (reuse OpenFold's) ------------------------------------
    data_module = OpenFoldDataModule(
        config=config.data,
        batch_seed=args.seed,
        **{k: v for k, v in vars(args).items() if k in [
            "train_data_dir", "train_alignment_dir", "template_mmcif_dir",
            "output_dir", "max_template_date", "train_mmcif_data_cache_path",
            "use_single_seq_mode", "distillation_data_dir",
            "distillation_alignment_dir", "val_data_dir", "val_alignment_dir",
            "val_mmcif_data_cache_path", "kalign_binary_path",
            "train_filter_path", "distillation_filter_path",
            "obsolete_pdbs_file_path", "template_release_dates_cache_path",
            "use_small_bfd", "train_chain_data_cache_path",
            "distillation_chain_data_cache_path", "train_epoch_len",
            "alignment_index_path", "distillation_alignment_index_path",
            "_distillation_structure_index_path",
        ]},
    )
    data_module.prepare_data()
    data_module.setup()

    # --- Callbacks ---------------------------------------------------------
    callbacks = []
    if args.checkpoint_every_epoch:
        callbacks.append(ModelCheckpoint(
            every_n_epochs=1,
            auto_insert_metric_name=False,
            save_top_k=-1,
        ))
    if args.early_stopping:
        callbacks.append(EarlyStoppingVerbose(
            monitor="val/loss",
            patience=args.patience,
            verbose=True,
            mode="min",
        ))
    if args.log_lr:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    # --- Logger ------------------------------------------------------------
    loggers = []
    if args.wandb:
        loggers.append(WandbLogger(
            name=args.experiment_name,
            save_dir=args.output_dir,
            id=args.wandb_id,
            project=args.wandb_project,
            entity=args.wandb_entity,
        ))

    # --- Strategy ----------------------------------------------------------
    if args.deepspeed_config_path is not None:
        strategy = DeepSpeedStrategy(config=args.deepspeed_config_path)
    elif args.gpus is not None and args.gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = "auto"

    # --- Trainer -----------------------------------------------------------
    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        num_nodes=args.num_nodes,
        precision=args.precision,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=args.num_sanity_val_steps,
        reload_dataloaders_every_n_epochs=args.reload_dataloaders_every_n_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    trainer.fit(model_module, datamodule=data_module)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def bool_type(s: str):
    if s.lower() in ("false", "f", "no", "n", "0"):
        return False
    if s.lower() in ("true", "t", "yes", "y", "1"):
        return True
    raise ValueError(f"Cannot interpret {s!r} as bool")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a retrieval-augmented fusion model on frozen OpenFold SoloSeq.",
    )

    # === Positional (same as train_openfold.py) ============================
    parser.add_argument("train_data_dir", type=str)
    parser.add_argument("train_alignment_dir", type=str)
    parser.add_argument("template_mmcif_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("max_template_date", type=str)

    # === Retrieval-specific ================================================
    parser.add_argument("--embedding_db_dir", type=str, required=True,
                        help="Directory of precomputed ESM-1b .pt files for the database")
    parser.add_argument("--openfold_checkpoint", type=str, default=None,
                        help="Path to a pretrained OpenFold SoloSeq checkpoint")
    parser.add_argument("--top_k", type=int, default=16,
                        help="Number of database entries to retrieve per query")
    parser.add_argument("--retriever_proj_dim", type=int, default=128,
                        help="Projection dimension for retriever scoring")
    parser.add_argument("--fusion_heads", type=int, default=8,
                        help="Number of attention heads in cross-attention fusion")
    parser.add_argument("--fusion_dropout", type=float, default=0.0,
                        help="Dropout in fusion cross-attention")
    parser.add_argument("--max_db_entries", type=int, default=-1,
                        help="Max entries to load from the database (-1 = all)")
    parser.add_argument("--db_max_seq_len", type=int, default=512,
                        help="Pad/truncate database embeddings to this length")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for retriever + fusion modules")

    # === Data (mirrored from train_openfold.py) ============================
    parser.add_argument("--config_preset", type=str, default="seqemb_initial_training")
    parser.add_argument("--experiment_config_json", type=str, default="")
    parser.add_argument("--train_mmcif_data_cache_path", type=str, default=None)
    parser.add_argument("--use_single_seq_mode", type=str, default=False)
    parser.add_argument("--distillation_data_dir", type=str, default=None)
    parser.add_argument("--distillation_alignment_dir", type=str, default=None)
    parser.add_argument("--val_data_dir", type=str, default=None)
    parser.add_argument("--val_alignment_dir", type=str, default=None)
    parser.add_argument("--val_mmcif_data_cache_path", type=str, default=None)
    parser.add_argument("--kalign_binary_path", type=str, default="/usr/bin/kalign")
    parser.add_argument("--train_filter_path", type=str, default=None)
    parser.add_argument("--distillation_filter_path", type=str, default=None)
    parser.add_argument("--obsolete_pdbs_file_path", type=str, default=None)
    parser.add_argument("--template_release_dates_cache_path", type=str, default=None)
    parser.add_argument("--use_small_bfd", type=bool_type, default=False)
    parser.add_argument("--train_chain_data_cache_path", type=str, default=None)
    parser.add_argument("--distillation_chain_data_cache_path", type=str, default=None)
    parser.add_argument("--train_epoch_len", type=int, default=10000)
    parser.add_argument("--alignment_index_path", type=str, default=None)
    parser.add_argument("--distillation_alignment_index_path", type=str, default=None)
    parser.add_argument("--_distillation_structure_index_path", type=str, default=None)

    # === Training ==========================================================
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--log_every_n_steps", type=int, default=25)
    parser.add_argument("--num_sanity_val_steps", type=int, default=0)
    parser.add_argument("--reload_dataloaders_every_n_epochs", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--checkpoint_every_epoch", action="store_true", default=False)
    parser.add_argument("--early_stopping", type=bool_type, default=False)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--deepspeed_config_path", type=str, default=None)
    parser.add_argument("--log_lr", action="store_true", default=False)

    # === Logging ===========================================================
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()
    main(args)
