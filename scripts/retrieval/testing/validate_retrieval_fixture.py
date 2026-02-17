#!/usr/bin/env python3
"""Validate retrieval fixture integrity from download_structures.py outputs.

Checks:
- required files exist
- FASTA IDs are unique
- split membership matches splits.json
- splits/FASTA IDs are mutually consistent
- structure files exist (supports .cif and .cif.gz)
"""

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple


def parse_fasta(path: Path) -> Dict[str, str]:
    records = {}
    header = None
    seq_chunks = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(seq_chunks)
                    if header in records:
                        raise ValueError(f"Duplicate FASTA ID in {path}: {header}")
                    records[header] = seq
                header = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)

    if header is not None:
        seq = "".join(seq_chunks)
        if header in records:
            raise ValueError(f"Duplicate FASTA ID in {path}: {header}")
        records[header] = seq

    return records


def _possible_structure_paths(structures_dir: Path, pdb_id: str) -> Iterable[Path]:
    pdb_id = pdb_id.lower()
    yield structures_dir / f"{pdb_id}.cif"
    yield structures_dir / f"{pdb_id}.cif.gz"


def _structure_exists(structures_dir: Path, pdb_id: str) -> Tuple[bool, str]:
    for p in _possible_structure_paths(structures_dir, pdb_id):
        if p.exists():
            return True, str(p)
    return False, ""


def _gunzip_smoke(path: Path) -> None:
    if path.suffix != ".gz":
        return
    with gzip.open(path, "rt") as f:
        _ = f.readline()


def validate(dataset_dir: Path, fail_fast: bool = False) -> int:
    splits_path = dataset_dir / "splits.json"
    train_fasta_path = dataset_dir / "train.fasta"
    val_fasta_path = dataset_dir / "val.fasta"
    structures_dir = dataset_dir / "structures"

    required = [splits_path, train_fasta_path, val_fasta_path, structures_dir]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("[ERROR] Missing required files/directories:")
        for m in missing:
            print(f"  - {m}")
        return 1

    splits = json.loads(splits_path.read_text())
    train_fasta = parse_fasta(train_fasta_path)
    val_fasta = parse_fasta(val_fasta_path)

    train_ids = set(train_fasta.keys())
    val_ids = set(val_fasta.keys())

    if train_ids & val_ids:
        overlap = sorted(train_ids & val_ids)
        print(f"[ERROR] Train/val FASTA overlap ({len(overlap)}): {overlap[:10]}")
        return 1

    split_train_ids = set()
    split_val_ids = set()
    if isinstance(splits, dict):
        split_train_ids = set(splits.get("train", []))
        split_val_ids = set(splits.get("val", []))
    elif isinstance(splits, list):
        for row in splits:
            if not isinstance(row, dict):
                continue
            split = row.get("split")
            if split not in {"train", "val"}:
                continue
            if "pdb_id" in row and "chain_id" in row:
                sid = f"{row['pdb_id']}_{row['chain_id']}"
            elif "id" in row:
                sid = str(row["id"])
            elif "sequence_id" in row:
                sid = str(row["sequence_id"])
            elif "domain_id" in row:
                sid = str(row["domain_id"])
            else:
                continue
            if split == "train":
                split_train_ids.add(sid)
            else:
                split_val_ids.add(sid)
    else:
        print(f"[ERROR] Unsupported splits.json format: {type(splits).__name__}")
        return 1

    errors = []

    if split_train_ids != train_ids:
        errors.append(
            f"train.fasta IDs ({len(train_ids)}) do not match splits.json train IDs ({len(split_train_ids)})"
        )
    if split_val_ids != val_ids:
        errors.append(
            f"val.fasta IDs ({len(val_ids)}) do not match splits.json val IDs ({len(split_val_ids)})"
        )

    fasta_ids = train_ids | val_ids

    for sid, seq in {**train_fasta, **val_fasta}.items():
        if not seq:
            errors.append(f"Empty sequence for {sid}")
            if fail_fast:
                break

    for sid in sorted(fasta_ids):
        pdb_id = sid[:4]
        ok, path = _structure_exists(structures_dir, pdb_id)
        if not ok:
            errors.append(f"Missing structure for {sid} (expected {pdb_id}.cif or .cif.gz)")
            if fail_fast:
                break
            continue
        _gunzip_smoke(Path(path))

    if errors:
        print(f"[ERROR] Validation failed with {len(errors)} issue(s):")
        for e in errors[:50]:
            print(f"  - {e}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")
        return 1

    print("[OK] Retrieval fixture validation passed")
    print(f"  train IDs: {len(train_ids)}")
    print(f"  val IDs  : {len(val_ids)}")
    print(f"  total    : {len(fasta_ids)}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--fail_fast", action="store_true")
    args = parser.parse_args()

    raise SystemExit(validate(args.dataset_dir, args.fail_fast))


if __name__ == "__main__":
    main()
