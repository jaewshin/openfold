import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import wandb
import logging
import argparse
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# this file is available on hugg
INPUT_FILE = "/scratch/akeluska/prot_distill_divide/tmvec2_pairs_predictions.parquet" 

WANDB_CONFIG = {
    "entity": None,
    "project": "tm-distill",
}

# Amino acid vocabulary
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWYXUBZ'
VOCAB_SIZE = len(AMINO_ACIDS) + 2
PAD_TOKEN = 0
UNK_TOKEN = 1
AA_TO_IDX = {aa: i + 2 for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX['<PAD>'] = PAD_TOKEN
AA_TO_IDX['<UNK>'] = UNK_TOKEN


# ==============================================================================
# SEQUENCE ENCODING
# ==============================================================================

def encode_sequence(sequence, max_length=None):
    """Encode amino acid sequence to tensor."""
    encoded = []
    for aa in sequence:
        encoded.append(AA_TO_IDX.get(aa, UNK_TOKEN))

    if max_length:
        if len(encoded) > max_length:
            encoded = encoded[:max_length]
        else:
            encoded.extend([PAD_TOKEN] * (max_length - len(encoded)))

    return torch.tensor(encoded, dtype=torch.long)


# ==============================================================================
# MODEL ARCHITECTURE (PURE ENCODER)
# ==============================================================================

class ProteinSequenceEncoder(nn.Module):
    """BiLSTM encoder with attention pooling."""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=512, output_dim=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim // 2, 2,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        mask = (x != PAD_TOKEN).float()
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)

        # Attention pooling
        attention_scores = self.attention(lstm_out).squeeze(-1)
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=1)

        pooled = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        output = self.output_proj(pooled)
        output = self.norm(output)

        return output


class StudentModel(nn.Module):
    """
    Prediction: CosineSimilarity(Enc(A), Enc(B))
    """

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=512,
                 seq_embed_dim=512, dropout=0.1):
        super().__init__()

        self.seq_encoder = ProteinSequenceEncoder(
            vocab_size, embed_dim, hidden_dim, seq_embed_dim, dropout
        )

    def forward(self, seq_a, seq_b=None):
        repr_a = self.seq_encoder(seq_a)

        if seq_b is not None:
            repr_b = self.seq_encoder(seq_b)

            # Calculate cosine similarity (-1 to 1)
            cosine_sim = F.cosine_similarity(repr_a, repr_b, dim=1)
            return repr_a, repr_b, cosine_sim
        else:
            return repr_a


# ==============================================================================
# DATASET
# ==============================================================================

class ProteinPairDataset(Dataset):
    """Dataset for protein pairs with TM-scores."""

    def __init__(self, df, max_length=600):
        self.df = df
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        seq_a = encode_sequence(row['seq_a'], self.max_length)
        seq_b = encode_sequence(row['seq_b'], self.max_length)
        tm_score = torch.tensor(row['tm_score'], dtype=torch.float32)

        return {
            'seq_a': seq_a,
            'seq_b': seq_b,
            'tm_score': tm_score
        }

# ==============================================================================
# LOSS FUNCTION (SIMPLIFIED)
# ==============================================================================

class CosineLoss(nn.Module):
    """
    Simple MSE Loss designed to force Cosine Sim to match TM Score.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        # targets are 0.0 to 1.0 (TM Score)
        # predictions are -1.0 to 1.0 (Cosine Sim)
        # We penalize the difference directly.
        # If prediction is negative but target is 0.1, it learns to be positive.
        loss = self.mse(predictions, targets)
        
        info = {
            'mse_loss': loss.item(),
            'pred_mean': predictions.mean().item(),
            'target_mean': targets.mean().item()
        }
        return loss, info

def get_metrics(true_scores, pred_scores):
    """Compute metrics (clamping predictions to valid 0-1 range first)."""
    true_scores = np.array(true_scores)
    pred_scores = np.clip(np.array(pred_scores), 0.0, 1.0)

    metrics = {}
    metrics['overall_r2'] = r2_score(true_scores, pred_scores)
    metrics['overall_mae'] = mean_absolute_error(true_scores, pred_scores)
    metrics['overall_mse'] = np.mean((true_scores - pred_scores) ** 2)
    metrics['bias'] = np.mean(pred_scores - true_scores)

    return metrics


# ==============================================================================
# PLOTTING
# ==============================================================================

def create_scatter_plot(true_scores, pred_scores, epoch, r2, plot_dir, save_name):
    plt.figure(figsize=(10, 10))
    
    # Clip for plotting only
    plot_preds = np.clip(pred_scores, 0.0, 1.0)

    plt.scatter(true_scores, plot_preds, alpha=0.5, s=8, color='purple')

    min_val = 0.0
    max_val = 1.0
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect')

    z = np.polyfit(true_scores, plot_preds, 1)
    p = np.poly1d(z)
    plt.plot(true_scores, p(true_scores), 'g-', linewidth=2, alpha=0.8, label=f'Fit')

    plt.xlabel('True TM-Score', fontsize=14, fontweight='bold')
    plt.ylabel('Cosine Similarity (Clipped 0-1)', fontsize=14, fontweight='bold')
    plt.title(f'Epoch {epoch+1}: Pure Cosine Model\nR² = {r2:.6f}',
              fontsize=14, fontweight='bold')

    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(plot_dir, f"{save_name}_epoch_{epoch+1:02d}_r2_{r2:.6f}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


# ==============================================================================
# TRAINING
# ==============================================================================

def train(
    data_path,
    batch_size=64,
    num_epochs=25,
    learning_rate=1e-3,
    device='cuda',
    max_length=600,
    train_split=0.85,
    max_samples=None
):
    config = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "model_type": "cosine_only"
    }

    wandb.init(
        entity=WANDB_CONFIG["entity"],
        project=WANDB_CONFIG["project"],
        config=config,
        reinit=True
    )

    plot_dir = "cosine_training_plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Load Data
    print(f"\nLoading dataset from: {data_path}")
    df = pd.read_parquet(data_path)

    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)

    train_size = int(len(df) * train_split)
    train_df = df[:train_size].copy()
    val_df = df[train_size:].copy()
    print(f"Training: {len(train_df):,}, Validation: {len(val_df):,}")

    train_dataset = ProteinPairDataset(train_df, max_length=max_length)
    val_dataset = ProteinPairDataset(val_df, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize Model
    model = StudentModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters (Encoder Only): {total_params:,}")

    # Loss - Using simplified MSE
    criterion = CosineLoss()
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=num_epochs)

    best_mae = float('inf')

    print(f"\nStarting training for {num_epochs} epochs...\n")

    for epoch in range(num_epochs):
        model.train()
        train_loss_accum = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in train_pbar:
            seq_a = batch['seq_a'].to(device)
            seq_b = batch['seq_b'].to(device)
            tm_score_true = batch['tm_score'].to(device)

            optimizer.zero_grad()
            
            # Forward returns cosine sim
            _, _, cosine_pred = model(seq_a, seq_b)

            loss, _ = criterion(cosine_pred, tm_score_true)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss_accum += loss.item()
            train_pbar.set_postfix({'MSE': f"{loss.item():.4f}"})

        # Validation
        model.eval()
        all_preds = []
        all_true = []

        with torch.no_grad():
            for batch in val_loader:
                seq_a = batch['seq_a'].to(device)
                seq_b = batch['seq_b'].to(device)
                tm_score_true = batch['tm_score']

                _, _, cosine_pred = model(seq_a, seq_b)

                all_preds.extend(cosine_pred.cpu().numpy())
                all_true.extend(tm_score_true.numpy())

        # Metrics (handles clipping internally)
        metrics = get_metrics(all_true, all_preds)
        
        # Plotting
        plot_path = create_scatter_plot(all_true, all_preds, epoch, metrics['overall_r2'], plot_dir, "cosine")

        if metrics['overall_mae'] < best_mae:
            best_mae = metrics['overall_mae']
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'config': config
            }, "cosine_student_best_tmvec2.pt")

        print(f"\nEpoch {epoch+1}: R²={metrics['overall_r2']:.4f}, MAE={metrics['overall_mae']:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss_accum / len(train_loader),
            **metrics
        })

    wandb.finish()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=INPUT_FILE)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--max-samples', type=int, default=None)
    args = parser.parse_args()

    train(args.data, args.batch_size, args.epochs, max_samples=args.max_samples)