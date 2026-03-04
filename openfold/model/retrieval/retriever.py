from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class LazyFaissRetriever:
    """Lazy CPU FAISS loader with reconstruct-based top-k retrieval."""

    def __init__(
        self,
        index_path: str,
        top_k: int = 8,
        nprobe: int = 64,
    ):
        self.index_path = index_path
        self.top_k = int(top_k)
        self.nprobe = int(nprobe)
        self._index = None
        self.dim: Optional[int] = None

    def _ensure_loaded(self):
        if self._index is not None:
            return

        try:
            import faiss
        except Exception as exc:
            raise RuntimeError(
                "faiss is required for retrieval. Install faiss-cpu/faiss-gpu in the active environment."
            ) from exc

        logger.info("Loading FAISS index from %s", self.index_path)
        self._index = faiss.read_index(self.index_path)
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = self.nprobe
        self.dim = int(self._index.d)
        logger.info("Loaded index (dim=%d, ntotal=%d)", self.dim, int(self._index.ntotal))

    def search(self, query_vec: np.ndarray, top_k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_loaded()
        assert self._index is not None
        assert self.dim is not None

        k = int(top_k or self.top_k)
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError(f"Query dim mismatch: got {q.shape[1]} expected {self.dim}")

        distances, indices = self._index.search(q, k)
        d = distances[0].astype(np.float32, copy=False)
        i = indices[0].astype(np.int64, copy=False)

        vectors = np.zeros((k, self.dim), dtype=np.float32)
        for j, idx in enumerate(i.tolist()):
            if idx < 0:
                continue
            try:
                vectors[j] = self._index.reconstruct(int(idx))
            except Exception:
                # Keep zero vector for entries that cannot be reconstructed.
                pass

        return d, i, vectors


def search_index(
    query_vec: torch.Tensor,
    retriever: LazyFaissRetriever,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Search FAISS and return differentiable scores over fixed top-k candidates."""
    q_np = query_vec.detach().float().cpu().numpy()
    _, indices_np, vectors_np = retriever.search(q_np, top_k=top_k)

    device = query_vec.device
    indices = torch.from_numpy(indices_np).to(device=device, dtype=torch.long)
    valid = indices >= 0
    retrieved_keys = torch.from_numpy(vectors_np).to(device=device, dtype=torch.float32)

    logits = torch.matmul(retrieved_keys, query_vec) / math.sqrt(max(1, query_vec.shape[-1]))
    if valid.any():
        logits = logits.masked_fill(~valid, -1e9)
        scores = torch.softmax(logits, dim=0)
    else:
        scores = torch.zeros_like(logits)

    return scores, indices, retrieved_keys, valid
