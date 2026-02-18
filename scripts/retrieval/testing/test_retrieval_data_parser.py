#!/usr/bin/env python3
"""Tests for retrieval dataset parsing utilities."""

import json
import sys
import tempfile
from pathlib import Path

# Allow running directly from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.retrieval.retrieval_data import (
    build_manifest,
    read_manifest_jsonl,
    write_manifest_jsonl,
)


def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_manifest_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_dir = root / "training_data"
        structures = data_dir / "structures"
        structures.mkdir(parents=True, exist_ok=True)

        _write_text(
            data_dir / "train.fasta",
            ">1abc_A\nACDE\n",
        )
        _write_text(
            data_dir / "val.fasta",
            ">2xyz_B\nGGTT\n",
        )
        (structures / "1abc.cif.gz").write_bytes(b"dummy")
        (structures / "2xyz.cif.gz").write_bytes(b"dummy")

        splits = [
            {"pdb_id": "1abc", "chain_id": "A", "split": "train", "domain_id": "1abcA00"},
            {"pdb_id": "2xyz", "chain_id": "B", "split": "val", "domain_id": "2xyzB00"},
        ]
        _write_text(data_dir / "splits.json", json.dumps(splits))

        records = build_manifest(data_dir, strict=True)
        assert len(records) == 2
        assert records[0].split == "train"
        assert records[1].split == "val"
        assert records[0].sequence_id == "1abc_A"
        assert records[1].sequence_id == "2xyz_B"

        manifest_path = data_dir / "manifest.jsonl"
        write_manifest_jsonl(records, manifest_path)
        restored = read_manifest_jsonl(manifest_path)
        assert len(restored) == 2
        assert restored[0].sequence == "ACDE"
        assert restored[1].sequence == "GGTT"


def main():
    test_manifest_roundtrip()
    print("[OK] retrieval_data parser tests passed")


if __name__ == "__main__":
    main()
