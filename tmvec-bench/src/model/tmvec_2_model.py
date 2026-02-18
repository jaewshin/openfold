"""TM-Vec model: Trainable Transformer encoder for TM-score prediction."""

from typing import Dict, Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR


class TMVecConfig:
    """Configuration for TM-Vec model."""

    def __init__(
        self,
        # Architecture
        d_model: int = 408,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.2,
        activation: str = 'gelu',
        out_dim: int = 512,
        projection_hidden_dim: Optional[int] = None,
        # Optimizer
        lr: float = 3e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        # Scheduler
        warmup_steps: int = 5698,
        max_steps: int = 113960,
        eta_min_factor: float = 0.1,
        restart_period: int = 22792,
        restart_mult: float = 1.0,
        # Training
        gradient_clip_val: float = 1.0,
    ):
        # Architecture
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.out_dim = out_dim
        self.projection_hidden_dim = projection_hidden_dim or d_model

        # Optimizer
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.eps = eps

        # Scheduler
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min_factor = eta_min_factor
        self.restart_period = restart_period
        self.restart_mult = restart_mult

        # Training
        self.gradient_clip_val = gradient_clip_val


class TMScorePredictor(pl.LightningModule):
    """
    TM-Vec: Predicts TM-scores from protein embedding pairs.

    Architecture:
        Embeddings → Transformer → Pool → MLP → Cosine Similarity → TM-score
    """

    def __init__(self, config: Optional[TMVecConfig] = None, **kwargs):
        super().__init__()

        # Config
        self.config = config if config else TMVecConfig(**kwargs)
        self.save_hyperparameters(vars(self.config))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation=self.config.activation,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)

        # Projection head
        self.dropout = nn.Dropout(self.config.dropout)
        self.projection = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.projection_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.projection_hidden_dim, self.config.out_dim)
        )

        # Loss
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.l1_loss = nn.L1Loss(reduction='mean')

        self._init_weights()
        self._print_architecture()

    def _init_weights(self):
        """Xavier uniform initialization for projection head."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _print_architecture(self):
        """Print model summary."""
        enc_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        proj_params = sum(p.numel() for p in self.projection.parameters() if p.requires_grad)

        print(f"✓ TM-Vec Model Architecture:")
        print(f"    Transformer: {self.config.num_layers} layers × {self.config.d_model}d × {self.config.nhead} heads")
        print(f"    Projection: {self.config.d_model} → {self.config.projection_hidden_dim} → {self.config.out_dim}")
        print(f"    Parameters: {enc_params:,} (encoder) + {proj_params:,} (projection) = {enc_params+proj_params:,} total")

    def encode_sequence(self, embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence: Transformer → Mean Pool → Project.

        Args:
            embeddings: [B, L, D]
            padding_mask: [B, L] True=padding

        Returns:
            [B, out_dim]
        """
        # Transformer
        hidden = self.encoder(embeddings, src_key_padding_mask=padding_mask)

        # Mean pooling (ignore padding)
        lengths = (~padding_mask).sum(dim=1, keepdim=True).float().clamp(min=1e-9)
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        pooled = (hidden * mask_expanded).sum(dim=1) / lengths

        # Project
        pooled = self.dropout(pooled)
        output = self.projection(pooled)

        return output

    def forward(
        self,
        seq1_embeddings: torch.Tensor,
        seq2_embeddings: torch.Tensor,
        seq1_padding_mask: torch.Tensor,
        seq2_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            (emb1, emb2, cosine_similarity)
        """
        emb1 = self.encode_sequence(seq1_embeddings, seq1_padding_mask)
        emb2 = self.encode_sequence(seq2_embeddings, seq2_padding_mask)
        cos_sim = self.cos_sim(emb1, emb2)

        return emb1, emb2, cos_sim

    def compute_loss(self, cos_sim: torch.Tensor, tm_targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """L1 loss between cosine similarity and TM-score."""
        loss = self.l1_loss(cos_sim, tm_targets)
        return loss, {'loss': loss, 'cos_sim_mean': cos_sim.mean().detach()}

    def compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor, prefix: str = '') -> Dict:
        """Compute MSE and accuracy metrics."""
        metrics = {f'{prefix}mse': F.mse_loss(preds, targets)}

        # Accuracy for similar structures (TM > 0.5)
        high_mask = targets >= 0.5
        if high_mask.sum() > 0:
            high_correct = (preds[high_mask] >= 0.5).sum().float()
            metrics[f'{prefix}acc_above_0.5'] = high_correct / high_mask.sum()

        # Accuracy for dissimilar structures (TM < 0.5)
        low_mask = targets < 0.5
        if low_mask.sum() > 0:
            low_correct = (preds[low_mask] < 0.5).sum().float()
            metrics[f'{prefix}acc_below_0.5'] = low_correct / low_mask.sum()

        return metrics

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        seq1_emb, seq1_mask, seq2_emb, seq2_mask, tm_targets = batch

        _, _, cos_sim = self(seq1_emb, seq2_emb, seq1_mask, seq2_mask)
        loss, _ = self.compute_loss(cos_sim, tm_targets)
        metrics = self.compute_metrics(cos_sim, tm_targets, prefix='train_')

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        seq1_emb, seq1_mask, seq2_emb, seq2_mask, tm_targets = batch

        _, _, cos_sim = self(seq1_emb, seq2_emb, seq1_mask, seq2_mask)
        loss, _ = self.compute_loss(cos_sim, tm_targets)
        metrics = self.compute_metrics(cos_sim, tm_targets, prefix='val_')

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        seq1_emb, seq1_mask, seq2_emb, seq2_mask, tm_targets = batch

        _, _, cos_sim = self(seq1_emb, seq2_emb, seq1_mask, seq2_mask)
        loss, _ = self.compute_loss(cos_sim, tm_targets)
        metrics = self.compute_metrics(cos_sim, tm_targets, prefix='test_')

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """AdamW with linear warmup + cosine annealing with warm restarts."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
            eps=self.config.eps
        )

        # Warmup: linear increase from 0.01×lr to lr
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_steps
        )

        # Main: cosine decay with periodic restarts
        cosine = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.restart_period,
            T_mult=int(self.config.restart_mult),
            eta_min=self.config.lr * self.config.eta_min_factor
        )

        # Combine: warmup → cosine restarts
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.config.warmup_steps]
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
