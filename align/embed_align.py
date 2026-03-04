from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_a.embeddings import build_embedder, l2_normalize_rows
from pipeline_a.io_utils import parse_single_fasta, write_jsonl


STATE_STOP = 0
STATE_M = 1
STATE_X = 2  # gap in candidate => deletion relative to query
STATE_Y = 3  # gap in query => insertion relative to query


@dataclass
class SWTrace:
    score: float
    q_start: int
    s_start: int
    q_end: int
    s_end: int
    ops: List[str]


@dataclass
class AlignmentResult:
    seq_id: str
    ann_score: float
    sequence: str
    sw_score: float
    q_start: int
    s_start: int
    q_end: int
    s_end: int
    ops: List[str]
    aligned_query_len: int
    matched_query_len: int
    query_coverage: float
    score_density: float
    gap_frac: float
    rank_score: float


def smith_waterman_affine(
    query_residue_embeddings: np.ndarray,
    candidate_residue_embeddings: np.ndarray,
    *,
    scale: float = 10.0,
    bias: float = 0.0,
    gap_open: float = -8.0,
    gap_extend: float = -0.5,
) -> SWTrace:
    """Local affine-gap SW with cosine-sim substitution."""
    n = int(query_residue_embeddings.shape[0])
    m = int(candidate_residue_embeddings.shape[0])

    if n == 0 or m == 0:
        return SWTrace(score=0.0, q_start=0, s_start=0, q_end=0, s_end=0, ops=[])

    qn = l2_normalize_rows(query_residue_embeddings.astype(np.float32))
    sn = l2_normalize_rows(candidate_residue_embeddings.astype(np.float32))
    sim = qn @ sn.T  # [n, m]

    M = np.zeros((n + 1, m + 1), dtype=np.float32)
    X = np.zeros((n + 1, m + 1), dtype=np.float32)
    Y = np.zeros((n + 1, m + 1), dtype=np.float32)

    ptr_M = np.zeros((n + 1, m + 1), dtype=np.uint8)
    ptr_X = np.zeros((n + 1, m + 1), dtype=np.uint8)
    ptr_Y = np.zeros((n + 1, m + 1), dtype=np.uint8)

    best_score = 0.0
    best_i = 0
    best_j = 0
    best_state = STATE_STOP

    go_ge = gap_open + gap_extend

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = scale * float(sim[i - 1, j - 1]) + bias

            prev = M[i - 1, j - 1]
            ptr = STATE_M
            if X[i - 1, j - 1] > prev:
                prev = X[i - 1, j - 1]
                ptr = STATE_X
            if Y[i - 1, j - 1] > prev:
                prev = Y[i - 1, j - 1]
                ptr = STATE_Y
            m_val = prev + sub
            if m_val <= 0.0:
                m_val = 0.0
                ptr = STATE_STOP
            M[i, j] = m_val
            ptr_M[i, j] = ptr

            x_val = M[i - 1, j] + go_ge
            x_ptr = STATE_M
            candidate = X[i - 1, j] + gap_extend
            if candidate > x_val:
                x_val = candidate
                x_ptr = STATE_X
            candidate = Y[i - 1, j] + go_ge
            if candidate > x_val:
                x_val = candidate
                x_ptr = STATE_Y
            if x_val <= 0.0:
                x_val = 0.0
                x_ptr = STATE_STOP
            X[i, j] = x_val
            ptr_X[i, j] = x_ptr

            y_val = M[i, j - 1] + go_ge
            y_ptr = STATE_M
            candidate = Y[i, j - 1] + gap_extend
            if candidate > y_val:
                y_val = candidate
                y_ptr = STATE_Y
            candidate = X[i, j - 1] + go_ge
            if candidate > y_val:
                y_val = candidate
                y_ptr = STATE_X
            if y_val <= 0.0:
                y_val = 0.0
                y_ptr = STATE_STOP
            Y[i, j] = y_val
            ptr_Y[i, j] = y_ptr

            if m_val > best_score:
                best_score = float(m_val)
                best_i, best_j, best_state = i, j, STATE_M
            if x_val > best_score:
                best_score = float(x_val)
                best_i, best_j, best_state = i, j, STATE_X
            if y_val > best_score:
                best_score = float(y_val)
                best_i, best_j, best_state = i, j, STATE_Y

    if best_state == STATE_STOP or best_score <= 0.0:
        return SWTrace(score=0.0, q_start=0, s_start=0, q_end=0, s_end=0, ops=[])

    i = best_i
    j = best_j
    state = best_state
    ops_rev: List[str] = []

    while state != STATE_STOP:
        if state == STATE_M:
            if i == 0 or j == 0 or M[i, j] <= 0.0:
                break
            ops_rev.append("M")
            state = int(ptr_M[i, j])
            i -= 1
            j -= 1
        elif state == STATE_X:
            if i == 0 or X[i, j] <= 0.0:
                break
            ops_rev.append("D")
            state = int(ptr_X[i, j])
            i -= 1
        elif state == STATE_Y:
            if j == 0 or Y[i, j] <= 0.0:
                break
            ops_rev.append("I")
            state = int(ptr_Y[i, j])
            j -= 1
        else:
            break

    ops = list(reversed(ops_rev))
    return SWTrace(
        score=best_score,
        q_start=i,
        s_start=j,
        q_end=best_i,
        s_end=best_j,
        ops=ops,
    )


def _alignment_stats(trace: SWTrace, query_length: int) -> Dict[str, float]:
    aligned_query_len = sum(1 for op in trace.ops if op in {"M", "D"})
    matched_query_len = sum(1 for op in trace.ops if op == "M")
    query_coverage = (aligned_query_len / query_length) if query_length > 0 else 0.0
    score_density = (trace.score / aligned_query_len) if aligned_query_len > 0 else 0.0

    # projected row has one aligned column per query residue. All non-match
    # columns are '-' in the aligned matrix view.
    gap_frac = 1.0 - ((matched_query_len / query_length) if query_length > 0 else 0.0)

    rank_score = (
        trace.score
        + 2.0 * aligned_query_len
        + 5.0 * math.log1p(max(score_density, 0.0))
    )

    return {
        "aligned_query_len": aligned_query_len,
        "matched_query_len": matched_query_len,
        "query_coverage": query_coverage,
        "score_density": score_density,
        "gap_frac": gap_frac,
        "rank_score": rank_score,
    }


def build_alignment_result(
    *,
    seq_id: str,
    ann_score: float,
    sequence: str,
    trace: SWTrace,
    query_length: int,
) -> AlignmentResult:
    stats = _alignment_stats(trace=trace, query_length=query_length)
    return AlignmentResult(
        seq_id=seq_id,
        ann_score=float(ann_score),
        sequence=sequence,
        sw_score=float(trace.score),
        q_start=trace.q_start,
        s_start=trace.s_start,
        q_end=trace.q_end,
        s_end=trace.s_end,
        ops=trace.ops,
        aligned_query_len=int(stats["aligned_query_len"]),
        matched_query_len=int(stats["matched_query_len"]),
        query_coverage=float(stats["query_coverage"]),
        score_density=float(stats["score_density"]),
        gap_frac=float(stats["gap_frac"]),
        rank_score=float(stats["rank_score"]),
    )


def align_candidate(
    *,
    seq_id: str,
    ann_score: float,
    query_sequence: str,
    candidate_sequence: str,
    embedder_name: str,
    embedder_device: str = "cpu",
    scale: float = 10.0,
    bias: float = 0.0,
    gap_open: float = -8.0,
    gap_extend: float = -0.5,
    embedder=None,
):
    if embedder is None:
        embedder = build_embedder(embedder_name, device=embedder_device)
    query_residues = embedder.embed_residues(query_sequence)
    candidate_residues = embedder.embed_residues(candidate_sequence)

    trace = smith_waterman_affine(
        query_residue_embeddings=query_residues,
        candidate_residue_embeddings=candidate_residues,
        scale=scale,
        bias=bias,
        gap_open=gap_open,
        gap_extend=gap_extend,
    )
    return build_alignment_result(
        seq_id=seq_id,
        ann_score=float(ann_score),
        sequence=candidate_sequence,
        trace=trace,
        query_length=len(query_sequence),
    )


def gate_and_rank(
    rows: Sequence[AlignmentResult],
    *,
    min_query_coverage: float = 0.15,
    min_aligned_query_len: int = 30,
    min_score_density: float = 1.0,
    max_gap_frac: float = 0.85,
    max_rows: int = 256,
) -> List[AlignmentResult]:
    survivors = [
        row
        for row in rows
        if row.query_coverage >= min_query_coverage
        and row.aligned_query_len >= min_aligned_query_len
        and row.score_density >= min_score_density
        and row.gap_frac <= max_gap_frac
    ]
    survivors.sort(key=lambda r: r.rank_score, reverse=True)
    return survivors[:max_rows]


def _load_candidates(path: str | Path) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _run_cli(args: argparse.Namespace) -> None:
    query_id, query_sequence = parse_single_fasta(args.query_fasta)
    candidates = _load_candidates(args.candidates_jsonl)
    embedder = build_embedder(args.embedder, device=args.device)

    rows: List[AlignmentResult] = []
    for cand in candidates:
        row = align_candidate(
            seq_id=cand["seq_id"],
            ann_score=float(cand["ann_score"]),
            query_sequence=query_sequence,
            candidate_sequence=cand["sequence"],
            embedder_name=args.embedder,
            embedder_device=args.device,
            scale=args.scale,
            bias=args.bias,
            gap_open=args.gap_open,
            gap_extend=args.gap_extend,
            embedder=embedder,
        )
        rows.append(row)

    gated = gate_and_rank(
        rows,
        min_query_coverage=args.min_query_coverage,
        min_aligned_query_len=args.min_aligned_query_len,
        min_score_density=args.min_score_density,
        max_gap_frac=args.max_gap_frac,
        max_rows=args.max_rows,
    )

    payload = [asdict(row) for row in gated]
    write_jsonl(args.output_jsonl, payload)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embedding-scored SW alignment module")
    parser.add_argument("--query_fasta", required=True)
    parser.add_argument("--candidates_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)

    parser.add_argument("--embedder", default="esm2_35m", choices=["esm2_35m", "aa_onehot"])
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--gap_open", type=float, default=-8.0)
    parser.add_argument("--gap_extend", type=float, default=-0.5)

    parser.add_argument("--min_query_coverage", type=float, default=0.15)
    parser.add_argument("--min_aligned_query_len", type=int, default=30)
    parser.add_argument("--min_score_density", type=float, default=1.0)
    parser.add_argument("--max_gap_frac", type=float, default=0.85)
    parser.add_argument("--max_rows", type=int, default=256)
    return parser


if __name__ == "__main__":
    _run_cli(_build_arg_parser().parse_args())
