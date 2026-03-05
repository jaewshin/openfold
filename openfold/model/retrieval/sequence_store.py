from __future__ import annotations

from collections import OrderedDict
from typing import Optional


class FastaSequenceStore:
    """ID->sequence lookup backed by Biopython SeqIO sqlite index."""

    def __init__(self, fasta_path: str, index_db_path: str, cache_size: int = 8192):
        try:
            from Bio import SeqIO
        except Exception as exc:
            raise RuntimeError("Biopython is required for raw-sequence retrieval pipeline.") from exc

        self.fasta_path = str(fasta_path)
        self.index_db_path = str(index_db_path)
        self.cache_size = int(cache_size)
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        self._records = SeqIO.index_db(self.index_db_path, [self.fasta_path], "fasta")

    def close(self):
        try:
            self._records.close()
        except Exception:
            pass

    def _cache_put(self, key: str, value: str):
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def get(self, seq_id: str) -> Optional[str]:
        cached = self._cache.get(seq_id)
        if cached is not None:
            self._cache.move_to_end(seq_id)
            return cached
        rec = self._records.get(seq_id)
        if rec is None:
            return None
        seq = str(rec.seq)
        self._cache_put(seq_id, seq)
        return seq

