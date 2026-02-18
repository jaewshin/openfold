#!/usr/bin/env python3
"""End-to-end OpenProteinNet bootstrap for retrieval training.

This orchestrates the same three stages used in this repo:
1) Download OpenProteinNet assets from s3://openfold
2) Prepare retrieval train/val manifests + FASTA
3) Generate per-sequence ESM-1b embeddings for train/val sequences

The script is resumable by default:
- Download step is skipped if requested subset directories already exist.
- Prepare step is skipped if expected output files already exist.
- Embedding step is skipped if .npy count matches train+val FASTA entry count.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Set


LOG = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: Sequence[str]) -> None:
    pretty = " ".join(cmd)
    LOG.info("Running: %s", pretty)
    subprocess.run(list(cmd), check=True)


def _count_fasta_records(path: Path) -> int:
    count = 0
    with open(path, "r") as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count


def _subset_dir_name(subset: str) -> str:
    mapping = {
        "pdb": "pdb",
        "uniclust30_filtered": "uniclust30",
        "data_caches": "data_caches",
        "pdb_mmcif": "pdb_mmcif",
    }
    if subset not in mapping:
        raise ValueError(f"Unsupported subset: {subset}")
    return mapping[subset]


def _has_any_file(root: Path) -> bool:
    if not root.exists():
        return False
    for p in root.rglob("*"):
        if p.is_file():
            return True
    return False


def _download_done(output_root: Path, subsets: Sequence[str]) -> bool:
    for s in subsets:
        subset_dir = output_root / _subset_dir_name(s)
        if not _has_any_file(subset_dir):
            return False
    return True


def _prepare_done(ready_dir: Path) -> bool:
    needed = [
        ready_dir / "manifest.jsonl",
        ready_dir / "train.fasta",
        ready_dir / "val.fasta",
        ready_dir / "openproteinnet_prepare_summary.json",
    ]
    return all(p.exists() for p in needed)


def _embedding_files(embedding_dir: Path) -> Set[str]:
    if not embedding_dir.exists():
        return set()
    out: Set[str] = set()
    for p in embedding_dir.glob("*.npy"):
        out.add(p.stem)
    return out


def _embedding_done(ready_dir: Path, embedding_dir: Path) -> bool:
    train_fasta = ready_dir / "train.fasta"
    val_fasta = ready_dir / "val.fasta"
    if not train_fasta.exists() or not val_fasta.exists():
        return False
    expected = _count_fasta_records(train_fasta) + _count_fasta_records(val_fasta)
    actual = len(_embedding_files(embedding_dir))
    LOG.info("Embedding completeness check: expected=%d actual=%d", expected, actual)
    return expected > 0 and expected == actual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download + prepare + embed OpenProteinNet for retrieval training",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Directory where OpenProteinNet subsets will be downloaded",
    )
    parser.add_argument(
        "--ready_dir",
        type=Path,
        default=None,
        help="Prepared dataset output dir (default: <output_root>/retrieval_ready_mmseqs)",
    )

    # Step toggles.
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--skip_embed", action="store_true")
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_prepare", action="store_true")
    parser.add_argument("--force_embed", action="store_true")

    # Download options.
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=["pdb", "uniclust30_filtered", "data_caches", "pdb_mmcif"],
        default=["pdb", "uniclust30_filtered", "data_caches"],
    )
    parser.add_argument("--download_workers", type=int, default=16)
    parser.add_argument(
        "--download_limit",
        type=int,
        default=0,
        help="Optional cap for smoke tests (0 disables)",
    )

    # Prepare options.
    parser.add_argument("--subset", type=str, default="uniclust30")
    parser.add_argument("--train_fraction", type=float, default=0.98)
    parser.add_argument("--min_length", type=int, default=16)
    parser.add_argument("--progress_every", type=int, default=5000)
    parser.add_argument(
        "--split_strategy",
        choices=["hash", "mmseqs"],
        default="mmseqs",
        help="Use mmseqs to reduce leakage by clustering homologs.",
    )
    parser.add_argument("--mmseqs_bin", type=str, default="mmseqs")
    parser.add_argument("--mmseqs_min_seq_id", type=float, default=0.3)
    parser.add_argument("--mmseqs_coverage", type=float, default=0.8)
    parser.add_argument("--mmseqs_cov_mode", type=int, default=0)
    parser.add_argument("--mmseqs_threads", type=int, default=16)
    parser.add_argument("--mmseqs_tmp_dir", type=Path, default=None)

    # Embedding options.
    parser.add_argument("--embedding_dir", type=Path, default=None)
    parser.add_argument("--embedding_device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--embedding_toks_per_batch", type=int, default=2048)
    parser.add_argument("--truncation_seq_length", type=int, default=1022)
    parser.add_argument("--embedding_max_sequences", type=int, default=0)

    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    output_root = args.output_root.resolve()
    ready_dir = (args.ready_dir or (output_root / "retrieval_ready_mmseqs")).resolve()
    embedding_dir = (args.embedding_dir or (ready_dir / "seq_embedding_esm1b")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ready_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    download_script = REPO_ROOT / "scripts/retrieval/download_openproteinnet.py"
    prepare_script = REPO_ROOT / "scripts/retrieval/prepare_openproteinnet_dataset.py"
    embed_script = REPO_ROOT / "scripts/retrieval/generate_esm1b_seq_embeddings.py"
    for script in (download_script, prepare_script, embed_script):
        if not script.exists():
            raise FileNotFoundError(f"Required script not found: {script}")

    if (
        not args.skip_prepare
        and args.split_strategy == "mmseqs"
        and shutil.which(args.mmseqs_bin) is None
        and "/" not in args.mmseqs_bin
    ):
        raise SystemExit(
            f"split_strategy=mmseqs requested but binary not found: {args.mmseqs_bin}. "
            "Install MMseqs2 or set --mmseqs_bin to an absolute path."
        )

    LOG.info("=== OpenProteinNet E2E bootstrap ===")
    LOG.info("output_root: %s", output_root)
    LOG.info("ready_dir: %s", ready_dir)
    LOG.info("embedding_dir: %s", embedding_dir)

    # Step 1: Download.
    if args.skip_download:
        LOG.info("Step 1/3 download: skipped (--skip_download)")
    else:
        done = _download_done(output_root, args.subsets)
        if done and not args.force_download:
            LOG.info("Step 1/3 download: already complete, skipping")
        else:
            cmd: List[str] = [
                sys.executable,
                str(download_script),
                "--output-dir",
                str(output_root),
                "--workers",
                str(args.download_workers),
                "--subsets",
                *args.subsets,
            ]
            if args.download_limit > 0:
                cmd.extend(["--limit", str(args.download_limit)])
            _run(cmd)

    # Step 2: Prepare retrieval dataset.
    if args.skip_prepare:
        LOG.info("Step 2/3 prepare: skipped (--skip_prepare)")
    else:
        done = _prepare_done(ready_dir)
        if done and not args.force_prepare:
            LOG.info("Step 2/3 prepare: already complete, skipping")
        else:
            cmd = [
                sys.executable,
                str(prepare_script),
                "--openproteinnet_dir",
                str(output_root),
                "--output_dir",
                str(ready_dir),
                "--subset",
                args.subset,
                "--train_fraction",
                str(args.train_fraction),
                "--min_length",
                str(args.min_length),
                "--progress_every",
                str(args.progress_every),
                "--split_strategy",
                args.split_strategy,
                "--mmseqs_bin",
                args.mmseqs_bin,
                "--mmseqs_min_seq_id",
                str(args.mmseqs_min_seq_id),
                "--mmseqs_coverage",
                str(args.mmseqs_coverage),
                "--mmseqs_cov_mode",
                str(args.mmseqs_cov_mode),
                "--mmseqs_threads",
                str(args.mmseqs_threads),
                "--structure_mode",
                "absolute",
            ]
            if args.mmseqs_tmp_dir is not None:
                cmd.extend(["--mmseqs_tmp_dir", str(args.mmseqs_tmp_dir)])
            _run(cmd)

    # Step 3: Generate ESM-1b query embeddings.
    if args.skip_embed:
        LOG.info("Step 3/3 embedding: skipped (--skip_embed)")
    else:
        done = _embedding_done(ready_dir, embedding_dir)
        if done and not args.force_embed:
            LOG.info("Step 3/3 embedding: already complete, skipping")
        else:
            cmd = [
                sys.executable,
                str(embed_script),
                "--dataset_dir",
                str(ready_dir),
                "--output_dir",
                str(embedding_dir),
                "--device",
                args.embedding_device,
                "--toks_per_batch",
                str(args.embedding_toks_per_batch),
                "--truncation_seq_length",
                str(args.truncation_seq_length),
            ]
            if args.embedding_max_sequences > 0:
                cmd.extend(["--max_sequences", str(args.embedding_max_sequences)])
            if args.force_embed:
                cmd.append("--overwrite")
            _run(cmd)

    LOG.info("E2E pipeline complete.")
    LOG.info("Prepared dataset: %s", ready_dir)
    LOG.info("Sequence embeddings: %s", embedding_dir)


if __name__ == "__main__":
    main()
