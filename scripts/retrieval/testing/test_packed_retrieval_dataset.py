#!/usr/bin/env python3
"""Tests for packed retrieval dataset format and loader."""

import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import Dataset

# Allow running directly from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.retrieval.retrieval_data import (
    PackedRetrievalDataset,
    RetrievalDataModule,
    TrainingRecord,
    pack_dataset_split,
)


class _DummyDataset(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Shapes mirror core OpenFold inputs enough for a smoke test.
        return {
            "seq_embedding": torch.randn(8, 16, 1, dtype=torch.float32),
            "seq_mask": torch.ones(8, 1, dtype=torch.float32),
            "aatype": torch.zeros(8, 1, dtype=torch.int64),
        }


def _make_records(n: int, split: str):
    return [
        TrainingRecord(
            sequence_id=f"{i:04d}_A",
            split=split,
            pdb_id=f"{i:04d}",
            chain_id="A",
            sequence="ACDE",
            structure_path="dummy.cif.gz",
        )
        for i in range(n)
    ]


def test_packed_dataset_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "packed"
        (root / "train").mkdir(parents=True, exist_ok=True)
        (root / "val").mkdir(parents=True, exist_ok=True)

        pack_dataset_split(
            dataset=_DummyDataset(5),
            records=_make_records(5, "train"),
            output_dir=root / "train",
            shard_size=2,
            float16_storage=True,
        )
        pack_dataset_split(
            dataset=_DummyDataset(2),
            records=_make_records(2, "val"),
            output_dir=root / "val",
            shard_size=2,
            float16_storage=True,
        )

        train_ds = PackedRetrievalDataset(root, split="train")
        assert len(train_ds) == 5
        sample = train_ds[0]
        assert "seq_embedding" in sample
        assert sample["seq_embedding"].dtype == torch.float32  # restored from fp16 storage

        dm = RetrievalDataModule(packed_dataset_dir=root, batch_size=2, num_workers=0)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert tuple(batch["seq_embedding"].shape[:2]) == (2, 8)


def main():
    test_packed_dataset_roundtrip()
    print("[OK] packed retrieval dataset tests passed")


if __name__ == "__main__":
    main()

