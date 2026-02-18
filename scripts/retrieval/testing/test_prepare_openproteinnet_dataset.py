#!/usr/bin/env python3
"""Tests for OpenProteinNet conversion utility."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Allow running directly from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PDB_STUB = """MODEL     1
ATOM      1  N   MET A   1      -1.000  -1.000  -1.000  1.00 50.00           N
ATOM      2  CA  MET A   1       0.000   0.000   0.000  1.00 50.00           C
ATOM      3  C   MET A   1       1.000   0.000   0.000  1.00 50.00           C
ATOM      4  N   GLY A   2       1.500   1.000   0.000  1.00 50.00           N
ATOM      5  CA  GLY A   2       2.500   1.000   0.000  1.00 50.00           C
TER
ENDMDL
END
"""


def _write_pdb(root: Path, entry: str) -> None:
    pdb_dir = root / "uniclust30" / entry / "pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    (pdb_dir / f"{entry}.pdb").write_text(PDB_STUB)


def test_prepare_openproteinnet_dataset_cli():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        input_root = tmp_root / "openproteinnet"
        out_root = tmp_root / "prepared"
        _write_pdb(input_root, "A0A000TEST1")
        _write_pdb(input_root, "A0A000TEST2")

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/retrieval/prepare_openproteinnet_dataset.py"),
            "--openproteinnet_dir",
            str(input_root),
            "--output_dir",
            str(out_root),
            "--min_length",
            "1",
            "--train_fraction",
            "0.5",
            "--progress_every",
            "1",
        ]
        subprocess.run(cmd, check=True)

        manifest_path = out_root / "manifest.jsonl"
        train_fasta = out_root / "train.fasta"
        val_fasta = out_root / "val.fasta"
        splits_path = out_root / "splits.json"
        summary_path = out_root / "openproteinnet_prepare_summary.json"

        assert manifest_path.exists()
        assert train_fasta.exists()
        assert val_fasta.exists()
        assert splits_path.exists()
        assert summary_path.exists()

        rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        assert len(rows) == 2
        assert all(r["chain_id"] == "A" for r in rows)
        assert all(r["sequence"] == "MG" for r in rows)
        assert all(Path(r["structure_path"]).is_absolute() for r in rows)

        splits = json.loads(splits_path.read_text())
        assert len(splits) == 2

        summary = json.loads(summary_path.read_text())
        assert summary["num_written"] == 2
        assert summary["num_train"] + summary["num_val"] == 2
