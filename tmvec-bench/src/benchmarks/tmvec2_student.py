#!/usr/bin/env python
"""TM-Vec Student: TM-score predictions for CATH and SCOPe using cosine similarity."""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.model.tmvec2_student_model import StudentModel, encode_sequence


def load_fasta(fasta_path, max_sequences):
    """Load sequences from FASTA file."""
    seq_ids, sequences = [], []
    current_id, current_seq = None, []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id:
                    seq_ids.append(current_id)
                    sequences.append("".join(current_seq))
                    if len(seq_ids) >= max_sequences:
                        break
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id and len(seq_ids) < max_sequences:
            seq_ids.append(current_id)
            sequences.append("".join(current_seq))

    print(f"Loaded {len(seq_ids)} sequences")
    return seq_ids, sequences


def load_model(checkpoint_path, device):
    """Load student model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))

    model = StudentModel()
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
    return model


def compute_embeddings(model, sequences, max_length, batch_size, device):
    """Encode sequences to embeddings."""
    print("Encoding sequences...")
    tokens = torch.stack([encode_sequence(seq, max_length) for seq in sequences])
    embeddings = []

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc="Encoding"):
            end = min(len(sequences), start + batch_size)
            batch = tokens[start:end].to(device)
            embeddings.append(model.seq_encoder(batch).cpu())

    return torch.cat(embeddings, dim=0)


def calculate_scores(embeddings):
    """Calculate pairwise TM-scores via cosine similarity.
    
    This follows the same approach as TMvec-2: normalize embeddings and
    compute cosine similarity as the TM-score prediction.
    """
    print("Calculating pairwise cosine similarities...")
    
    # Normalize embeddings (L2 normalization)
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    
    # Compute pairwise cosine similarity matrix
    # This is equivalent to normalized dot product
    tm_matrix = torch.mm(embeddings_norm, embeddings_norm.t()).cpu().numpy()
    
    print(f"Cosine similarity stats - Mean: {tm_matrix.mean():.4f}, Std: {tm_matrix.std():.4f}")
    
    return tm_matrix


def save_results(seq_ids, tm_matrix, output_path):
    """Save pairwise scores to CSV."""
    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    seq1_ids = []
    seq2_ids = []
    tm_scores = []

    # Only save upper triangle (i < j) to match other methods
    n = len(seq_ids)
    for i in range(n):
        for j in range(i + 1, n):
            seq1_ids.append(seq_ids[i])
            seq2_ids.append(seq_ids[j])
            tm_scores.append(float(tm_matrix[i, j]))

    df = pd.DataFrame({
        'seq1_id': seq1_ids,
        'seq2_id': seq2_ids,
        'tm_score': tm_scores
    })
    df.to_csv(output_path, index=False)
    print(f"Saved {len(tm_scores):,} scores")


def main():
    is_scope40 = len(sys.argv) > 1 and sys.argv[1] == "scope40"

    if is_scope40:
        fasta = "data/fasta/scope40-1000.fa"
        output = "results/scope40_tmvec2_student_similarities.csv"
        max_seq = 1000
    else:
        fasta = "data/cath-top1k.fa"
        output = "results/cath_tmvec2_student_similarities.csv"
        max_seq = 1000

    checkpoint = "binaries/tmvec2_student.pt"
    max_length = 600
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"FASTA: {fasta}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output: {output}")

    seq_ids, sequences = load_fasta(fasta, max_seq)
    model = load_model(checkpoint, device)
    embeddings = compute_embeddings(model, sequences, max_length, batch_size, device)
    tm_matrix = calculate_scores(embeddings)
    save_results(seq_ids, tm_matrix, Path(output))


if __name__ == "__main__":
    main()
