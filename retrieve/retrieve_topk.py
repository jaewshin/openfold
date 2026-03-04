from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_a.embeddings import build_embedder, l2_normalize_rows
from pipeline_a.io_utils import parse_single_fasta, write_jsonl
from pipeline_a.sequence_store import load_sequences_for_ids


@dataclass
class RetrievedCandidate:
    row: int
    seq_id: str
    ann_score: float
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)


class IdTextLookup:
    """Random access lookup for row->ID in a newline-delimited text file."""

    def __init__(self, ids_path: str | Path, offsets_path: str | Path | None = None):
        self.ids_path = Path(ids_path)
        self.offsets_path = Path(offsets_path) if offsets_path else None

        if self.offsets_path is not None and self.offsets_path.exists():
            self.offsets = np.memmap(self.offsets_path, mode="r", dtype=np.uint64)
            self.lines = None
            self._handle = open(self.ids_path, "rb")
        else:
            with open(self.ids_path, "r", encoding="utf-8") as handle:
                self.lines = [line.rstrip("\n") for line in handle]
            self.offsets = None
            self._handle = None

    def __del__(self):
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass

    def __len__(self) -> int:
        if self.offsets is not None:
            return int(self.offsets.shape[0])
        return len(self.lines)

    def get(self, row_index: int) -> str:
        if self.offsets is not None:
            self._handle.seek(int(self.offsets[row_index]))
            return self._handle.readline().decode("utf-8").rstrip("\n")
        return self.lines[row_index]


class FaissSearcher:
    def __init__(self, index_path: str | Path):
        try:
            import faiss
        except Exception as exc:  # pragma: no cover - dependency guard
            raise ImportError("FAISS is required for retrieve_topk.py") from exc

        self.faiss = faiss
        self.index = faiss.read_index(str(index_path))

    @property
    def dim(self) -> int:
        return int(self.index.d)

    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.index.d:
            raise ValueError(
                f"Query embedding dim {q.shape[1]} does not match index dim {self.index.d}"
            )
        distances, indices = self.index.search(q, int(top_k))
        return distances[0], indices[0]


def _select_top_k(
    *,
    searcher: FaissSearcher,
    id_lookup: IdTextLookup,
    query_embedding: np.ndarray,
    top_k: int,
) -> List[Tuple[int, str, float]]:
    distances, indices = searcher.search(query_embedding, top_k)
    out = []
    for dist, idx in zip(distances.tolist(), indices.tolist()):
        if idx < 0:
            continue
        seq_id = id_lookup.get(int(idx))
        out.append((int(idx), seq_id, float(dist)))
    return out


def _cheap_filter_and_dedup(
    *,
    query_length: int,
    rows: Sequence[Tuple[int, str, float]],
    sequence_map: Mapping[str, str],
    length_ratio_low: float,
    length_ratio_high: float,
    top_k_prime: int,
) -> List[RetrievedCandidate]:
    min_len = int(max(1, np.floor(query_length * length_ratio_low)))
    max_len = int(max(min_len, np.ceil(query_length * length_ratio_high)))

    seen_sequences = set()
    out: List[RetrievedCandidate] = []

    for row_idx, seq_id, ann_score in rows:
        sequence = sequence_map.get(seq_id)
        if sequence is None:
            continue

        seq_len = len(sequence)
        if seq_len < min_len or seq_len > max_len:
            continue

        if sequence in seen_sequences:
            continue
        seen_sequences.add(sequence)

        out.append(
            RetrievedCandidate(
                row=row_idx,
                seq_id=seq_id,
                ann_score=ann_score,
                sequence=sequence,
            )
        )

        if len(out) >= top_k_prime:
            break

    return out


def retrieve_topk_candidates(
    *,
    index_path: str | Path,
    ids_path: str | Path,
    ids_offsets_path: str | Path | None,
    query_embedding: np.ndarray,
    query_length: int,
    sequence_fasta_path: str | Path | None,
    sequence_sqlite_path: str | Path | None,
    build_sequence_sqlite_if_missing: bool,
    top_k: int,
    top_k_prime: int,
    length_ratio_low: float,
    length_ratio_high: float,
) -> List[RetrievedCandidate]:
    searcher = FaissSearcher(index_path=index_path)
    id_lookup = IdTextLookup(ids_path=ids_path, offsets_path=ids_offsets_path)

    top_rows = _select_top_k(
        searcher=searcher,
        id_lookup=id_lookup,
        query_embedding=query_embedding,
        top_k=top_k,
    )

    ids_to_fetch = [seq_id for _, seq_id, _ in top_rows]
    records = load_sequences_for_ids(
        seq_ids=ids_to_fetch,
        fasta_path=sequence_fasta_path,
        sqlite_path=sequence_sqlite_path,
        build_sqlite_if_missing=build_sequence_sqlite_if_missing,
    )
    sequence_map = {seq_id: rec.sequence for seq_id, rec in records.items()}

    return _cheap_filter_and_dedup(
        query_length=query_length,
        rows=top_rows,
        sequence_map=sequence_map,
        length_ratio_low=length_ratio_low,
        length_ratio_high=length_ratio_high,
        top_k_prime=top_k_prime,
    )


def _load_or_compute_query_embedding(args: argparse.Namespace, query_sequence: str) -> np.ndarray:
    if args.query_embedding_path:
        embedding = np.load(args.query_embedding_path)
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    else:
        embedder = build_embedder(args.embedder, device=args.device)
        embedding = embedder.embed_sequence(query_sequence).astype(np.float32)

    if args.normalize_query:
        embedding = l2_normalize_rows(embedding.reshape(1, -1))[0]

    return embedding


def _run_cli(args: argparse.Namespace) -> None:
    query_id, query_sequence = parse_single_fasta(args.query_fasta)
    query_embedding = _load_or_compute_query_embedding(args, query_sequence)

    candidates = retrieve_topk_candidates(
        index_path=args.index_path,
        ids_path=args.ids_path,
        ids_offsets_path=args.ids_offsets_path,
        query_embedding=query_embedding,
        query_length=len(query_sequence),
        sequence_fasta_path=args.sequence_fasta,
        sequence_sqlite_path=args.sequence_sqlite,
        build_sequence_sqlite_if_missing=args.build_sequence_sqlite_if_missing,
        top_k=args.top_k,
        top_k_prime=args.top_k_prime,
        length_ratio_low=args.length_ratio_low,
        length_ratio_high=args.length_ratio_high,
    )

    rows = [dataclasses.asdict(c) for c in candidates]
    write_jsonl(args.output_jsonl, rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve top-K candidates from a FAISS index")
    parser.add_argument("--query_fasta", required=True)

    parser.add_argument("--index_path", required=True)
    parser.add_argument("--ids_path", required=True)
    parser.add_argument("--ids_offsets_path", default=None)

    parser.add_argument("--sequence_fasta", default=None)
    parser.add_argument("--sequence_sqlite", default=None)
    parser.add_argument("--build_sequence_sqlite_if_missing", action="store_true", default=False)

    parser.add_argument("--query_embedding_path", default=None)
    parser.add_argument("--embedder", default="esm2_35m", choices=["esm2_35m", "aa_onehot"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--normalize_query", action="store_true", default=True)
    parser.add_argument("--no_normalize_query", dest="normalize_query", action="store_false")

    parser.add_argument("--top_k", type=int, default=50_000)
    parser.add_argument("--top_k_prime", type=int, default=2_000)
    parser.add_argument("--length_ratio_low", type=float, default=0.7)
    parser.add_argument("--length_ratio_high", type=float, default=1.3)

    parser.add_argument("--output_jsonl", required=True)
    return parser


if __name__ == "__main__":
    _run_cli(_build_arg_parser().parse_args())
