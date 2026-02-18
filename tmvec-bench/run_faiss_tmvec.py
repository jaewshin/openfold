#!/usr/bin/env python3
"""
Compute TMVec-2s (student) embeddings for sequences in a FASTA, using tmvec-bench codepaths:
  - StudentModel
  - encode_sequence
  - model.seq_encoder(tokens)

This mirrors src/benchmarks/tmvec2_student.py but streams FASTA and writes shards.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Iterator, Tuple, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Import EXACTLY like the benchmark does
from src.model.tmvec2_student_model import StudentModel, encode_sequence  # :contentReference[oaicite:2]{index=2}

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


def open_text_maybe_gz(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def iter_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """Yield (id, sequence) from FASTA (supports .gz)."""
    with open_text_maybe_gz(path) as f:
        header: Optional[str] = None
        seq_chunks: List[str] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, "".join(seq_chunks)


def load_tmvec2s_student(checkpoint_path: str, device: torch.device) -> StudentModel:
    """
    Loads weights the same way as tmvec-bench benchmark script:
      - checkpoint can be a dict containing model_state_dict / state_dict
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt

    model = StudentModel()
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


@torch.inference_mode()
def embed_batch(
    model: StudentModel,
    seqs: List[str],
    max_length: int,
    device: torch.device,
    amp: bool,
    normalize: bool,
) -> np.ndarray:
    # tokenization matches repo encode_sequence (pads/truncates to max_length) :contentReference[oaicite:3]{index=3}
    toks = torch.stack([encode_sequence(s, max_length) for s in seqs]).to(device)

    if device.type == "cuda" and amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            emb = model.seq_encoder(toks)
    else:
        emb = model.seq_encoder(toks)

    emb = emb.float()
    if normalize:
        emb = F.normalize(emb, p=2, dim=1)  # benchmark normalizes before cosine similarity :contentReference[oaicite:4]{index=4}

    return emb.cpu().numpy().astype(np.float32, copy=False)


def write_shard(out_dir: Path, shard_idx: int, ids: List[str], embs: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"part_{shard_idx:05d}.npy"
    ids_path = out_dir / f"part_{shard_idx:05d}.ids.txt"

    np.save(npy_path, embs)
    with open(ids_path, "w") as f:
        for _id in ids:
            f.write(_id + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True, help="Input FASTA (.fa / .fasta / .gz)")
    ap.add_argument("--out_dir", required=True, help="Output directory for shards")

    ap.add_argument("--checkpoint", default="", help="Path to tmvec2_student.pt (optional)")
    ap.add_argument("--download_ckpt", action="store_true", help="Download ckpt from HF if not provided")
    ap.add_argument("--hf_repo", default="scikit-bio/TMVec-2s", help="HF repo for TMVec-2s")  # :contentReference[oaicite:5]{index=5}
    ap.add_argument("--hf_filename", default="tmvec2_student.pt", help="HF filename for ckpt")  # :contentReference[oaicite:6]{index=6}

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--max_length", type=int, default=600)   # benchmark uses 600 :contentReference[oaicite:7]{index=7}
    ap.add_argument("--amp", action="store_true", help="Use CUDA autocast")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize embeddings (recommended for IP/cosine)")

    ap.add_argument("--shard_size", type=int, default=200_000, help="Sequences per output shard")

    args = ap.parse_args()

    device = torch.device(args.device if args.device != "cuda" else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)

    # Resolve checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        if not args.download_ckpt:
            raise SystemExit(
                "No --checkpoint provided.\n"
                "Either pass --checkpoint /path/to/tmvec2_student.pt\n"
                "or use --download_ckpt to fetch from Hugging Face."
            )
        if hf_hub_download is None:
            raise SystemExit("huggingface-hub is not installed. pip install huggingface-hub")
        ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_filename)

    print(f"[info] device={device}")
    print(f"[info] checkpoint={ckpt_path}")
    print(f"[info] fasta={args.fasta}")
    print(f"[info] out_dir={out_dir}")
    print(f"[info] max_length={args.max_length} batch_size={args.batch_size} shard_size={args.shard_size}")

    model = load_tmvec2s_student(ckpt_path, device)

    shard_ids: List[str] = []
    shard_embs: List[np.ndarray] = []
    shard_count = 0
    shard_idx = 0

    batch_ids: List[str] = []
    batch_seqs: List[str] = []

    for seq_id, seq in tqdm(iter_fasta(args.fasta), desc="Embedding"):
        batch_ids.append(seq_id)
        batch_seqs.append(seq)

        if len(batch_seqs) >= args.batch_size:
            embs = embed_batch(model, batch_seqs, args.max_length, device, args.amp, args.normalize)

            shard_ids.extend(batch_ids)
            shard_embs.append(embs)
            shard_count += embs.shape[0]

            batch_ids.clear()
            batch_seqs.clear()

            if shard_count >= args.shard_size:
                out_embs = np.concatenate(shard_embs, axis=0)
                write_shard(out_dir, shard_idx, shard_ids, out_embs)
                print(f"[info] wrote shard {shard_idx:05d}: n={len(shard_ids)} shape={out_embs.shape}")
                shard_idx += 1
                shard_ids.clear()
                shard_embs.clear()
                shard_count = 0

    # tail
    if batch_seqs:
        embs = embed_batch(model, batch_seqs, args.max_length, device, args.amp, args.normalize)
        shard_ids.extend(batch_ids)
        shard_embs.append(embs)
        shard_count += embs.shape[0]

    if shard_ids:
        out_embs = np.concatenate(shard_embs, axis=0)
        write_shard(out_dir, shard_idx, shard_ids, out_embs)
        print(f"[info] wrote shard {shard_idx:05d}: n={len(shard_ids)} shape={out_embs.shape}")

    print("[done] embeddings complete.")


if __name__ == "__main__":
    main()
