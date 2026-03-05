from __future__ import annotations

import gzip
import logging
import shutil
import struct
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RowIdLookup:
    """Random-access row-id lookup from a line-based ids file."""

    def __init__(self, ids_path: str, cache_size: int = 8192):
        self.original_path = Path(ids_path)
        self.cache_size = int(cache_size)
        self._cache: "OrderedDict[int, str]" = OrderedDict()

        if self.original_path.suffix == ".pkl":
            txt_candidate = self.original_path.with_suffix(".txt")
            if not txt_candidate.exists():
                raise ValueError(
                    f"Unsupported ids mapping format for random access: {self.original_path}. "
                    "Provide a line-based .txt/.txt.gz map (one sequence id per line, row-aligned with FAISS)."
                )
            self.text_path = txt_candidate
        elif self.original_path.suffix == ".gz":
            text_path = self.original_path.with_suffix("")
            if not text_path.exists():
                logger.info("Decompressing ids map %s -> %s", self.original_path, text_path)
                with gzip.open(self.original_path, "rb") as src, open(text_path, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            self.text_path = text_path
        else:
            self.text_path = self.original_path

        if not self.text_path.exists():
            raise FileNotFoundError(f"IDs text path not found: {self.text_path}")

        self.offsets_path = self.text_path.with_suffix(self.text_path.suffix + ".offsets.u64")
        if not self.offsets_path.exists():
            self._build_offsets()

        self._offsets = np.memmap(self.offsets_path, mode="r", dtype=np.uint64)
        self._fh = open(self.text_path, "rb", buffering=1024 * 1024)

    def _build_offsets(self):
        logger.info("Building ids offset table: %s", self.offsets_path)
        with open(self.text_path, "rb", buffering=1024 * 1024) as f, open(self.offsets_path, "wb") as out:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                out.write(struct.pack("<Q", pos))

    def __len__(self) -> int:
        return int(self._offsets.shape[0])

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass

    def _cache_put(self, idx: int, value: str):
        self._cache[idx] = value
        self._cache.move_to_end(idx)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def get(self, row_idx: int) -> Optional[str]:
        row_idx = int(row_idx)
        if row_idx < 0 or row_idx >= len(self):
            return None
        cached = self._cache.get(row_idx)
        if cached is not None:
            self._cache.move_to_end(row_idx)
            return cached
        offset = int(self._offsets[row_idx])
        self._fh.seek(offset)
        line = self._fh.readline().decode("utf-8", errors="replace").strip()
        if not line:
            return None
        value = line.split()[0]
        self._cache_put(row_idx, value)
        return value

