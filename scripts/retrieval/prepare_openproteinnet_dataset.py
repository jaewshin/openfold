#!/usr/bin/env python3
"""Convert downloaded OpenProteinNet PDB files into retrieval training format.

Input layout (download_openproteinnet.py output):
  <openproteinnet_dir>/uniclust30/<entry_id>/pdb/<entry_id>.pdb

Output layout (compatible with retrieval_data.py + train_retrieval_lightning.py):
  <output_dir>/manifest.jsonl
  <output_dir>/train.fasta
  <output_dir>/val.fasta
  <output_dir>/splits.json
  <output_dir>/openproteinnet_prepare_summary.json

Notes:
  - This script extracts sequence directly from ATOM/HETATM CA records.
  - Default split is deterministic grouped hash train/val assignment.
  - For leakage-aware splits, use --split_strategy mmseqs to split by
    sequence-homology clusters rather than individual records.
  - By default, manifest stores absolute structure paths (no file copying).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


LOGGER = logging.getLogger(__name__)

AA3_TO_1: Dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "C",
    "PYL": "K",
    "ASX": "B",
    "GLX": "Z",
    "UNK": "X",
}


@dataclass
class CandidateRecord:
    entry_id: str
    pdb_id: str
    chain_id: str
    sequence: str
    structure_source_path: Path
    sequence_id: str
    split_group: str


def is_pdb_path(path: Path) -> bool:
    suffixes = {s.lower() for s in path.suffixes}
    return ".pdb" in suffixes or ".ent" in suffixes


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt")
    return open(path, "r")


def fasta_write(handle, header: str, sequence: str, width: int = 80) -> None:
    handle.write(f">{header}\n")
    for i in range(0, len(sequence), width):
        handle.write(sequence[i:i + width] + "\n")


def iter_uniclust_pdbs(openproteinnet_dir: Path, subset: str = "uniclust30") -> Iterator[Tuple[str, Path]]:
    subset_dir = openproteinnet_dir / subset
    if not subset_dir.exists():
        raise FileNotFoundError(f"Subset directory not found: {subset_dir}")

    for entry_dir in subset_dir.iterdir():
        if not entry_dir.is_dir():
            continue
        pdb_dir = entry_dir / "pdb"
        if not pdb_dir.exists():
            continue

        candidates = [p for p in pdb_dir.iterdir() if p.is_file() and is_pdb_path(p)]
        if not candidates:
            continue
        candidates.sort()
        yield entry_dir.name, candidates[0]


def extract_sequence_from_pdb(path: Path) -> Tuple[str, str]:
    """Extract first-chain amino-acid sequence from PDB ATOM/HETATM records."""
    first_chain = None
    seen_residues = set()
    seq: List[str] = []

    with open_text(path) as handle:
        for line in handle:
            if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue

            chain_id = line[21].strip() or "A"
            if first_chain is None:
                first_chain = chain_id
            if chain_id != first_chain:
                continue

            # Residue identity key: (chain, seq_id, insertion_code)
            res_key = (chain_id, line[22:26], line[26:27])
            if res_key in seen_residues:
                continue
            seen_residues.add(res_key)

            res_name = line[17:20].strip().upper()
            seq.append(AA3_TO_1.get(res_name, "X"))

    return (first_chain or "A"), "".join(seq)


def assign_split_from_key(group_key: str, train_fraction: float, seed: int) -> str:
    key = f"{seed}:{group_key}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)
    return "train" if value < train_fraction else "val"


def build_mmseqs_cluster_map(
    candidates: List[CandidateRecord],
    mmseqs_bin: str,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    threads: int,
    tmp_parent: Path,
) -> Dict[str, str]:
    resolved_mmseqs = shutil.which(mmseqs_bin) if "/" not in mmseqs_bin else mmseqs_bin
    if not resolved_mmseqs:
        raise RuntimeError(
            f"split_strategy=mmseqs requested but mmseqs binary not found: {mmseqs_bin}. "
            "Install MMseqs2 or pass --mmseqs_bin with a valid path."
        )

    tmp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="openproteinnet_mmseqs_", dir=str(tmp_parent)) as tmp:
        tmp_dir = Path(tmp)
        input_fasta = tmp_dir / "sequences.fasta"
        output_prefix = tmp_dir / "cluster_out"
        work_dir = tmp_dir / "work"
        tsv_path = Path(f"{output_prefix}_cluster.tsv")

        id_to_group: Dict[str, str] = {}
        with open(input_fasta, "w") as fh:
            for i, c in enumerate(candidates):
                sid = f"s{i}"
                id_to_group[sid] = c.split_group
                fasta_write(fh, sid, c.sequence)

        cmd = [
            resolved_mmseqs,
            "easy-cluster",
            str(input_fasta),
            str(output_prefix),
            str(work_dir),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            str(cov_mode),
            "--threads",
            str(threads),
        ]
        LOGGER.info("Running MMseqs2 clustering for leakage-aware split...")
        LOGGER.info("Command: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

        if not tsv_path.exists():
            raise RuntimeError(f"MMseqs2 output missing cluster TSV: {tsv_path}")

        group_to_rep: Dict[str, str] = {}
        with open(tsv_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rep_id, member_id = line.split("\t")
                member_group = id_to_group.get(member_id)
                rep_group = id_to_group.get(rep_id)
                if member_group is None or rep_group is None:
                    continue
                # Assign by split-group; duplicates share the same group id.
                group_to_rep[member_group] = rep_group

        # Ensure every group has a cluster key, even if MMseqs omits singletons.
        for g in id_to_group.values():
            group_to_rep.setdefault(g, g)

        LOGGER.info("MMseqs2 clusters assigned for %d groups", len(group_to_rep))
        return group_to_rep


def collect_candidates(
    openproteinnet_dir: Path,
    subset: str,
    min_length: int,
    max_samples: int,
) -> Tuple[List[CandidateRecord], Dict[str, int]]:
    used_sequence_ids: Dict[str, int] = {}
    counters = {
        "num_scanned": 0,
        "num_written": 0,
        "num_skipped_short": 0,
        "num_skipped_empty": 0,
        "num_parse_errors": 0,
    }
    records: List[CandidateRecord] = []

    for entry_id, pdb_path in iter_uniclust_pdbs(openproteinnet_dir, subset=subset):
        if max_samples and counters["num_written"] >= max_samples:
            break
        counters["num_scanned"] += 1

        try:
            chain_id, seq = extract_sequence_from_pdb(pdb_path)
        except Exception as exc:
            counters["num_parse_errors"] += 1
            LOGGER.warning("Failed to parse %s: %s", pdb_path, exc)
            continue

        if not seq:
            counters["num_skipped_empty"] += 1
            continue
        if len(seq) < min_length:
            counters["num_skipped_short"] += 1
            continue

        pdb_id = entry_id.lower()
        split_group = f"{pdb_id}_{chain_id}"
        sequence_id = split_group
        dup_count = used_sequence_ids.get(sequence_id, 0)
        used_sequence_ids[sequence_id] = dup_count + 1
        if dup_count > 0:
            sequence_id = f"{sequence_id}_dup{dup_count}"

        records.append(
            CandidateRecord(
                entry_id=entry_id,
                pdb_id=pdb_id,
                chain_id=chain_id,
                sequence=seq,
                structure_source_path=pdb_path,
                sequence_id=sequence_id,
                split_group=split_group,
            )
        )
        counters["num_written"] += 1

    return records, counters


def resolve_structure_path(
    source_path: Path,
    output_dir: Path,
    pdb_id: str,
    mode: str,
    used_dest_paths: Dict[str, int],
) -> Path:
    if mode == "absolute":
        return source_path.resolve()

    structures_dir = output_dir / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    base = f"{pdb_id}.pdb"
    count = used_dest_paths.get(base, 0)
    used_dest_paths[base] = count + 1
    name = base if count == 0 else f"{pdb_id}_dup{count}.pdb"
    dest = structures_dir / name

    if mode == "symlink":
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(source_path.resolve())
    elif mode == "copy":
        shutil.copy2(source_path, dest)
    else:
        raise ValueError(f"Unsupported structure mode: {mode}")

    return dest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare OpenProteinNet uniclust30 PDBs for retrieval training.",
    )
    parser.add_argument(
        "--openproteinnet_dir",
        type=Path,
        required=True,
        help="Directory containing uniclust30/ from download_openproteinnet.py",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for manifest/train.fasta/val.fasta",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="uniclust30",
        help="Subdirectory under openproteinnet_dir to parse (default: uniclust30)",
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.98,
        help="Deterministic train split fraction in [0,1] (default: 0.98)",
    )
    parser.add_argument(
        "--split_strategy",
        choices=["hash", "mmseqs"],
        default="hash",
        help=(
            "Split assignment strategy. "
            "'hash' is deterministic grouped hashing by sequence group; "
            "'mmseqs' clusters homologs and assigns clusters to split to reduce leakage."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used in deterministic split hashing")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional cap for smoke tests")
    parser.add_argument("--min_length", type=int, default=16, help="Skip sequences shorter than this")
    parser.add_argument(
        "--progress_every",
        type=int,
        default=1000,
        help="Emit progress log every N written records (default: 1000)",
    )
    parser.add_argument(
        "--structure_mode",
        choices=["absolute", "symlink", "copy"],
        default="absolute",
        help="How structure_path is materialized in output (default: absolute)",
    )
    parser.add_argument(
        "--skip_splits_json",
        action="store_true",
        help="Do not write splits.json (manifest + FASTA are still written)",
    )
    parser.add_argument("--mmseqs_bin", type=str, default="mmseqs", help="MMseqs2 binary for split_strategy=mmseqs")
    parser.add_argument("--mmseqs_min_seq_id", type=float, default=0.3, help="MMseqs min sequence identity")
    parser.add_argument("--mmseqs_coverage", type=float, default=0.8, help="MMseqs cluster coverage threshold")
    parser.add_argument("--mmseqs_cov_mode", type=int, default=0, help="MMseqs coverage mode")
    parser.add_argument("--mmseqs_threads", type=int, default=16, help="MMseqs threads")
    parser.add_argument(
        "--mmseqs_tmp_dir",
        type=Path,
        default=None,
        help="Optional parent directory for MMseqs temporary files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.train_fraction < 1.0):
        raise SystemExit("--train_fraction must be in (0, 1)")
    if args.max_samples < 0:
        raise SystemExit("--max_samples must be >= 0")
    if args.min_length < 1:
        raise SystemExit("--min_length must be >= 1")
    if args.progress_every < 1:
        raise SystemExit("--progress_every must be >= 1")
    if args.mmseqs_min_seq_id <= 0 or args.mmseqs_min_seq_id > 1:
        raise SystemExit("--mmseqs_min_seq_id must be in (0, 1]")
    if args.mmseqs_coverage <= 0 or args.mmseqs_coverage > 1:
        raise SystemExit("--mmseqs_coverage must be in (0, 1]")
    if args.mmseqs_threads < 1:
        raise SystemExit("--mmseqs_threads must be >= 1")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    train_fasta_path = output_dir / "train.fasta"
    val_fasta_path = output_dir / "val.fasta"
    splits_path = output_dir / "splits.json"
    summary_path = output_dir / "openproteinnet_prepare_summary.json"

    splits_rows: List[Dict[str, object]] = []
    used_dest_paths: Dict[str, int] = {}

    LOGGER.info("Scanning OpenProteinNet subset: %s", args.openproteinnet_dir / args.subset)
    candidates, counters = collect_candidates(
        openproteinnet_dir=args.openproteinnet_dir,
        subset=args.subset,
        min_length=args.min_length,
        max_samples=args.max_samples,
    )
    LOGGER.info(
        "Candidate collection done: kept=%d scanned=%d skipped_short=%d skipped_empty=%d parse_errors=%d",
        counters["num_written"],
        counters["num_scanned"],
        counters["num_skipped_short"],
        counters["num_skipped_empty"],
        counters["num_parse_errors"],
    )

    if args.split_strategy == "hash":
        split_by_group = {
            c.split_group: assign_split_from_key(c.split_group, args.train_fraction, args.seed)
            for c in candidates
        }
    else:
        mmseqs_tmp_parent = args.mmseqs_tmp_dir or (output_dir / ".tmp_mmseqs")
        group_to_rep = build_mmseqs_cluster_map(
            candidates=candidates,
            mmseqs_bin=args.mmseqs_bin,
            min_seq_id=args.mmseqs_min_seq_id,
            coverage=args.mmseqs_coverage,
            cov_mode=args.mmseqs_cov_mode,
            threads=args.mmseqs_threads,
            tmp_parent=mmseqs_tmp_parent,
        )
        split_by_group = {
            g: assign_split_from_key(rep, args.train_fraction, args.seed)
            for g, rep in group_to_rep.items()
        }

    num_train = 0
    num_val = 0
    with open(manifest_path, "w") as manifest_fh, open(train_fasta_path, "w") as train_fh, open(val_fasta_path, "w") as val_fh:
        for i, c in enumerate(candidates, start=1):
            split = split_by_group.get(c.split_group)
            if split is None:
                split = assign_split_from_key(c.split_group, args.train_fraction, args.seed)

            structure_path = resolve_structure_path(
                source_path=c.structure_source_path,
                output_dir=output_dir,
                pdb_id=c.pdb_id,
                mode=args.structure_mode,
                used_dest_paths=used_dest_paths,
            )

            row = {
                "sequence_id": c.sequence_id,
                "split": split,
                "pdb_id": c.pdb_id,
                "chain_id": c.chain_id,
                "sequence": c.sequence,
                "structure_path": str(structure_path),
                "domain_id": c.entry_id,
                "cath_code": "",
                "topology_code": "",
                "superfamily_code": "",
                "n_residues": len(c.sequence),
            }
            manifest_fh.write(json.dumps(row) + "\n")

            if split == "train":
                num_train += 1
                fasta_write(train_fh, c.sequence_id, c.sequence)
            else:
                num_val += 1
                fasta_write(val_fh, c.sequence_id, c.sequence)

            splits_rows.append(
                {
                    "split": split,
                    "pdb_id": c.pdb_id,
                    "chain_id": c.chain_id,
                    "domain_id": c.entry_id,
                    "cath_code": "",
                    "topology_code": "",
                    "superfamily_code": "",
                    "n_residues": len(c.sequence),
                }
            )

            if i % args.progress_every == 0:
                manifest_fh.flush()
                train_fh.flush()
                val_fh.flush()
                LOGGER.info(
                    "Write progress: written=%d train=%d val=%d",
                    i, num_train, num_val,
                )

    if not args.skip_splits_json:
        with open(splits_path, "w") as fh:
            json.dump(splits_rows, fh)

    summary = {
        "openproteinnet_dir": str(args.openproteinnet_dir),
        "subset": args.subset,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "train_fasta": str(train_fasta_path),
        "val_fasta": str(val_fasta_path),
        "splits_json": None if args.skip_splits_json else str(splits_path),
        "num_scanned": counters["num_scanned"],
        "num_written": counters["num_written"],
        "num_train": num_train,
        "num_val": num_val,
        "num_skipped_short": counters["num_skipped_short"],
        "num_skipped_empty": counters["num_skipped_empty"],
        "num_parse_errors": counters["num_parse_errors"],
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "split_strategy": args.split_strategy,
        "structure_mode": args.structure_mode,
        "max_samples": args.max_samples,
        "min_length": args.min_length,
        "mmseqs": {
            "mmseqs_bin": args.mmseqs_bin,
            "min_seq_id": args.mmseqs_min_seq_id,
            "coverage": args.mmseqs_coverage,
            "cov_mode": args.mmseqs_cov_mode,
            "threads": args.mmseqs_threads,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
