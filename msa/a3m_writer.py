from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openfold.data import parsers

from pipeline_a.io_utils import ensure_parent_dir


@dataclass
class A3MProjectedRow:
    seq_id: str
    header: str
    sequence: str
    aligned_query_length: int


def project_trace_to_a3m_row(
    *,
    query_length: int,
    candidate_sequence: str,
    q_start: int,
    s_start: int,
    ops: Sequence[str],
    seq_id: str,
    header: str,
    drop_trailing_insertions: bool = True,
) -> A3MProjectedRow:
    row_cols = ["-"] * query_length
    insertions = [""] * (query_length + 1)

    qi = int(q_start)
    sj = int(s_start)

    for op in ops:
        if op == "M":
            if qi < query_length and sj < len(candidate_sequence):
                row_cols[qi] = candidate_sequence[sj].upper()
            qi += 1
            sj += 1
        elif op == "D":
            if qi < query_length:
                row_cols[qi] = "-"
            qi += 1
        elif op == "I":
            if sj < len(candidate_sequence):
                ins = candidate_sequence[sj].lower()
                if qi <= query_length:
                    insertions[qi] += ins
            sj += 1
        else:
            raise ValueError(f"Unsupported op '{op}'")

    parts: List[str] = []
    for i in range(query_length):
        parts.append(insertions[i])
        parts.append(row_cols[i])

    if not drop_trailing_insertions:
        parts.append(insertions[query_length])

    seq = "".join(parts)
    aligned_seq = seq.translate(str.maketrans("", "", "abcdefghijklmnopqrstuvwxyz"))
    if len(aligned_seq) != query_length:
        raise ValueError(
            f"Invalid A3M row for {seq_id}: aligned length {len(aligned_seq)} != query length {query_length}"
        )

    return A3MProjectedRow(
        seq_id=seq_id,
        header=header,
        sequence=seq,
        aligned_query_length=query_length,
    )


def write_a3m(
    *,
    output_path: str | Path,
    query_id: str,
    query_sequence: str,
    rows: Sequence[A3MProjectedRow],
) -> None:
    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f">{query_id}\n")
        handle.write(f"{query_sequence}\n")
        for row in rows:
            handle.write(f">{row.header}\n")
            handle.write(f"{row.sequence}\n")


def write_query_only_stockholm(
    *,
    output_path: str | Path,
    query_name: str,
    query_sequence: str,
) -> None:
    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("# STOCKHOLM 1.0\n")
        handle.write(f"{query_name}    {query_sequence}\n")
        handle.write("//\n")


def write_empty_hhr(output_path: str | Path) -> None:
    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8"):
        pass


def write_openfold_alignment_dir(
    *,
    alignments_root: str | Path,
    target_id: str,
    query_sequence: str,
    a3m_rows: Sequence[A3MProjectedRow],
) -> Path:
    target_dir = Path(alignments_root) / target_id
    target_dir.mkdir(parents=True, exist_ok=True)

    write_a3m(
        output_path=target_dir / "bfd_uniclust_hits.a3m",
        query_id=target_id,
        query_sequence=query_sequence,
        rows=a3m_rows,
    )
    write_query_only_stockholm(
        output_path=target_dir / "uniref90_hits.sto",
        query_name="query",
        query_sequence=query_sequence,
    )
    write_query_only_stockholm(
        output_path=target_dir / "mgnify_hits.sto",
        query_name="query",
        query_sequence=query_sequence,
    )
    write_empty_hhr(target_dir / "hhsearch_output.hhr")

    return target_dir


def validate_a3m_invariants(*, a3m_path: str | Path, query_sequence: str) -> Mapping[str, int]:
    with open(a3m_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    parsed = parsers.parse_a3m(content)
    if not parsed.sequences:
        raise ValueError("A3M parser produced zero sequences")
    if parsed.sequences[0] != query_sequence:
        raise ValueError("A3M first aligned sequence does not equal query sequence")

    q_len = len(query_sequence)
    for i, seq in enumerate(parsed.sequences):
        if len(seq) != q_len:
            raise ValueError(f"Aligned sequence {i} length {len(seq)} != query length {q_len}")

    if len(parsed.deletion_matrix) != len(parsed.sequences):
        raise ValueError("Deletion matrix row count does not match sequence count")

    for i, row in enumerate(parsed.deletion_matrix):
        if len(row) != q_len:
            raise ValueError(
                f"Deletion matrix row {i} length {len(row)} != query length {q_len}"
            )

    return {
        "num_sequences": len(parsed.sequences),
        "query_length": q_len,
    }
