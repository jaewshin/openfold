from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_a.embeddings import build_embedder
from pipeline_a.io_utils import iter_fasta


def _count_records(fasta_path: str | Path) -> int:
    return sum(1 for _ in iter_fasta(fasta_path))


def build_db_embeddings(
    *,
    fasta_path: str | Path,
    output_embeddings_path: str | Path,
    output_ids_path: str | Path,
    output_metadata_jsonl: str | Path | None,
    embedder_name: str,
    device: str,
    dtype: str,
    log_every: int,
) -> None:
    embedder = build_embedder(embedder_name, device=device)

    num_records = _count_records(fasta_path)
    if num_records == 0:
        raise ValueError(f"No FASTA records found in {fasta_path}")

    # Probe embedding dim with first sequence.
    first_id, first_desc, first_seq = next(iter(iter_fasta(fasta_path)))
    first_vec = embedder.embed_sequence(first_seq).astype(np.float32)
    dim = int(first_vec.shape[0])

    output_embeddings_path = Path(output_embeddings_path)
    output_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    output_ids_path = Path(output_ids_path)
    output_ids_path.parent.mkdir(parents=True, exist_ok=True)

    np_dtype = np.float16 if dtype == "fp16" else np.float32
    mmap = np.lib.format.open_memmap(
        output_embeddings_path,
        mode="w+",
        dtype=np_dtype,
        shape=(num_records, dim),
    )

    meta_handle = None
    if output_metadata_jsonl:
        output_metadata_jsonl = Path(output_metadata_jsonl)
        output_metadata_jsonl.parent.mkdir(parents=True, exist_ok=True)
        meta_handle = open(output_metadata_jsonl, "w", encoding="utf-8")

    try:
        with open(output_ids_path, "w", encoding="utf-8") as ids_handle:
            for idx, (seq_id, description, sequence) in enumerate(iter_fasta(fasta_path)):
                if idx == 0:
                    vec = first_vec
                else:
                    vec = embedder.embed_sequence(sequence).astype(np.float32)
                mmap[idx] = vec.astype(np_dtype)
                ids_handle.write(seq_id)
                ids_handle.write("\n")

                if meta_handle is not None:
                    row = {
                        "seq_id": seq_id,
                        "description": description,
                        "length": len(sequence),
                    }
                    meta_handle.write(json.dumps(row, sort_keys=True))
                    meta_handle.write("\n")

                if (idx + 1) % log_every == 0:
                    print(f"embedded {idx + 1}/{num_records}")
    finally:
        if meta_handle is not None:
            meta_handle.close()

    mmap.flush()
    print(f"Wrote embeddings: {output_embeddings_path}")
    print(f"Wrote ids: {output_ids_path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute sequence-level embeddings for a FASTA DB")
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output_embeddings", required=True)
    parser.add_argument("--output_ids", required=True)
    parser.add_argument("--output_metadata_jsonl", default=None)

    parser.add_argument("--embedder", default="esm2_35m", choices=["esm2_35m", "aa_onehot"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--log_every", type=int, default=500)
    return parser


def _run_cli(args: argparse.Namespace) -> None:
    build_db_embeddings(
        fasta_path=args.fasta,
        output_embeddings_path=args.output_embeddings,
        output_ids_path=args.output_ids,
        output_metadata_jsonl=args.output_metadata_jsonl,
        embedder_name=args.embedder,
        device=args.device,
        dtype=args.dtype,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    _run_cli(_build_arg_parser().parse_args())
