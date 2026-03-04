#!/usr/bin/env python3
"""Download and validate OpenProteinNet alignments for validation sequences.

This script:
1) Reads validation IDs/sequences from retrieval_ready/val.fasta
2) Maps each validation sequence ID to an OpenProteinNet entry via manifest.jsonl
3) Downloads precomputed alignment files from the public OpenFold S3 bucket:
     uniclust30/<ENTRY_ID>/a3m/uniclust30.a3m
   (falls back to .a3m.gz when needed)
4) Verifies each downloaded alignment corresponds to the validation sequence by
   comparing the first A3M sequence to the validation FASTA sequence.

Output summary is written to JSON and printed to stdout.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

LOGGER = logging.getLogger("download_openproteinnet_val_alignments")


def parse_fasta(path: Path) -> Dict[str, str]:
    records: Dict[str, List[str]] = {}
    current_id: str | None = None
    with path.open("r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_id = line[1:].split()[0]
                records.setdefault(current_id, [])
            else:
                if current_id is None:
                    raise ValueError(f"Invalid FASTA format in {path}: sequence before header")
                records[current_id].append(line)

    return {k: "".join(v).upper() for k, v in records.items()}


def infer_entry_id_from_seq_id(sequence_id: str) -> str:
    return sequence_id.split("_", 1)[0].upper()


def parse_val_entry_map(manifest_path: Path) -> Dict[str, str]:
    seq_to_entry: Dict[str, str] = {}
    with manifest_path.open("r") as fh:
        for raw in fh:
            row = json.loads(raw)
            if str(row.get("split", "")) != "val":
                continue

            seq_id = str(row.get("sequence_id", "")).strip()
            if not seq_id:
                continue

            entry_id = str(row.get("domain_id", "")).strip().upper()
            if not entry_id:
                structure_path = str(row.get("structure_path", "")).strip()
                if structure_path:
                    entry_id = Path(structure_path).parent.parent.name.upper()
            if not entry_id:
                entry_id = infer_entry_id_from_seq_id(seq_id)

            prev = seq_to_entry.get(seq_id)
            if prev is not None and prev != entry_id:
                raise ValueError(
                    f"Conflicting entry IDs for {seq_id}: {prev!r} vs {entry_id!r}"
                )
            seq_to_entry[seq_id] = entry_id

    return seq_to_entry


def normalize_alignment_sequence(seq: str) -> str:
    return "".join(ch for ch in seq if ch.isalpha()).upper()


def read_first_a3m_record(path: Path) -> Tuple[str, str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        saw_header = False
        header = ""
        chunks: List[str] = []
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if not saw_header:
                    saw_header = True
                    header = line[1:].strip()
                    continue
                if chunks:
                    break
                continue
            if saw_header:
                chunks.append(line)

    if not chunks:
        raise ValueError(f"No sequence data found in {path}")

    first_seq = normalize_alignment_sequence("".join(chunks))
    if not first_seq:
        raise ValueError(f"First sequence is empty after normalization: {path}")
    return header, first_seq


def is_subsequence_with_wildcard(query: str, target: str, wildcard: str = "X") -> bool:
    if not query:
        return True
    q_idx = 0
    for t_char in target:
        if q_idx >= len(query):
            break
        q_char = query[q_idx]
        if q_char == wildcard or t_char == wildcard or q_char == t_char:
            q_idx += 1
    return q_idx == len(query)


def download_url(url: str, dest_path: Path, timeout_sec: float) -> int:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    with urlopen(url, timeout=timeout_sec) as response, tmp_path.open("wb") as out_fh:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_fh.write(chunk)

    size = tmp_path.stat().st_size
    if size == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded empty file from {url}")

    tmp_path.replace(dest_path)
    return size


@dataclass(frozen=True)
class DownloadResult:
    entry_id: str
    status: str
    path: str | None = None
    bytes: int = 0
    error: str | None = None


def resolve_existing_alignment_path(openproteinnet_dir: Path, subset: str, entry_id: str) -> Path | None:
    base = openproteinnet_dir / subset / entry_id / "a3m"
    a3m = base / "uniclust30.a3m"
    a3m_gz = base / "uniclust30.a3m.gz"
    if a3m.exists() and a3m.stat().st_size > 0:
        return a3m
    if a3m_gz.exists() and a3m_gz.stat().st_size > 0:
        return a3m_gz
    return None


def download_entry_alignment(
    *,
    entry_id: str,
    openproteinnet_dir: Path,
    subset: str,
    base_url: str,
    timeout_sec: float,
    force: bool,
) -> DownloadResult:
    existing = resolve_existing_alignment_path(openproteinnet_dir, subset, entry_id)
    if existing is not None and not force:
        return DownloadResult(entry_id=entry_id, status="exists", path=str(existing))

    base_rel = f"{subset}/{entry_id}/a3m"
    targets = [
        ("uniclust30.a3m", openproteinnet_dir / base_rel / "uniclust30.a3m"),
        ("uniclust30.a3m.gz", openproteinnet_dir / base_rel / "uniclust30.a3m.gz"),
    ]

    errors: List[str] = []
    for filename, dest in targets:
        url = f"{base_url.rstrip('/')}/{base_rel}/{filename}"
        try:
            num_bytes = download_url(url, dest, timeout_sec=timeout_sec)
            return DownloadResult(
                entry_id=entry_id,
                status="downloaded",
                path=str(dest),
                bytes=num_bytes,
            )
        except HTTPError as exc:
            if exc.code == 404:
                errors.append(f"{filename}:404")
                continue
            errors.append(f"{filename}:HTTP{exc.code}")
        except URLError as exc:
            errors.append(f"{filename}:URLError:{exc.reason}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{filename}:{type(exc).__name__}:{exc}")

    return DownloadResult(
        entry_id=entry_id,
        status="failed",
        error="; ".join(errors) if errors else "unknown_error",
    )


def verify_alignments(
    *,
    val_sequences: Dict[str, str],
    seq_to_entry: Dict[str, str],
    openproteinnet_dir: Path,
    subset: str,
) -> Dict[str, object]:
    missing_alignment: List[str] = []
    parse_errors: List[Dict[str, str]] = []
    header_mismatches: List[Dict[str, str]] = []
    mismatches: List[Dict[str, object]] = []
    exact_matches = 0
    subsequence_matches = 0

    for seq_id, seq in val_sequences.items():
        entry_id = seq_to_entry.get(seq_id, infer_entry_id_from_seq_id(seq_id))
        aln_path = resolve_existing_alignment_path(openproteinnet_dir, subset, entry_id)
        if aln_path is None:
            missing_alignment.append(seq_id)
            continue

        try:
            aln_header, aln_query = read_first_a3m_record(aln_path)
        except Exception as exc:  # noqa: BLE001
            parse_errors.append(
                {
                    "sequence_id": seq_id,
                    "entry_id": entry_id,
                    "alignment_path": str(aln_path),
                    "error": str(exc),
                }
            )
            continue

        if entry_id.upper() not in aln_header.upper():
            header_mismatches.append(
                {
                    "sequence_id": seq_id,
                    "entry_id": entry_id,
                    "alignment_path": str(aln_path),
                    "observed_header": aln_header,
                }
            )
            continue

        expected = normalize_alignment_sequence(seq)
        if aln_query != expected:
            if is_subsequence_with_wildcard(expected, aln_query) or is_subsequence_with_wildcard(aln_query, expected):
                subsequence_matches += 1
            else:
                mismatches.append(
                    {
                        "sequence_id": seq_id,
                        "entry_id": entry_id,
                        "alignment_path": str(aln_path),
                        "expected_length": len(expected),
                        "observed_length": len(aln_query),
                    }
                )
        else:
            exact_matches += 1

    matched = exact_matches + subsequence_matches

    return {
        "num_sequences": len(val_sequences),
        "num_exact_matches": exact_matches,
        "num_subsequence_matches": subsequence_matches,
        "num_sequence_compatible": matched,
        "num_matched": matched,
        "num_missing_alignment": len(missing_alignment),
        "num_parse_errors": len(parse_errors),
        "num_header_mismatches": len(header_mismatches),
        "num_mismatches": len(mismatches),
        "missing_alignment_examples": missing_alignment[:20],
        "parse_error_examples": parse_errors[:20],
        "header_mismatch_examples": header_mismatches[:20],
        "mismatch_examples": mismatches[:20],
    }


def batch_download(
    *,
    entry_ids: Iterable[str],
    openproteinnet_dir: Path,
    subset: str,
    base_url: str,
    timeout_sec: float,
    workers: int,
    force: bool,
) -> Tuple[List[DownloadResult], int]:
    results: List[DownloadResult] = []
    total_bytes = 0
    unique_entries = sorted(set(entry_ids))
    if not unique_entries:
        return results, total_bytes

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                download_entry_alignment,
                entry_id=entry_id,
                openproteinnet_dir=openproteinnet_dir,
                subset=subset,
                base_url=base_url,
                timeout_sec=timeout_sec,
                force=force,
            ): entry_id
            for entry_id in unique_entries
        }
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            total_bytes += res.bytes
            if len(results) % 100 == 0 or len(results) == len(unique_entries):
                LOGGER.info("Download progress: %d/%d", len(results), len(unique_entries))

    results.sort(key=lambda r: r.entry_id)
    return results, total_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download + validate OpenProteinNet validation alignments from public OpenFold S3."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet/retrieval_ready"),
        help="Directory containing manifest.jsonl and val.fasta.",
    )
    parser.add_argument(
        "--openproteinnet-dir",
        type=Path,
        default=Path("/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet"),
        help="Root OpenProteinNet directory where uniclust30/<entry>/a3m/* will be stored.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="uniclust30",
        help="OpenProteinNet subset prefix (default: uniclust30).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://openfold.s3.amazonaws.com",
        help="Base URL for OpenFold public S3 bucket.",
    )
    parser.add_argument("--workers", type=int, default=16, help="Parallel download workers.")
    parser.add_argument("--timeout-sec", type=float, default=60.0, help="Per-request timeout in seconds.")
    parser.add_argument("--force", action="store_true", help="Re-download alignments even if files exist.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download and run validation only.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional JSON summary path (default: <dataset-dir>/openproteinnet_val_alignment_summary.json).",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    dataset_dir = args.dataset_dir
    openproteinnet_dir = args.openproteinnet_dir
    val_fasta = dataset_dir / "val.fasta"
    manifest_jsonl = dataset_dir / "manifest.jsonl"

    for p in (val_fasta, manifest_jsonl):
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    LOGGER.info("Loading validation sequences from %s", val_fasta)
    val_sequences = parse_fasta(val_fasta)
    LOGGER.info("Validation sequences: %d", len(val_sequences))

    LOGGER.info("Loading validation mapping from %s", manifest_jsonl)
    seq_to_entry = parse_val_entry_map(manifest_jsonl)
    missing_mapping = sorted(set(val_sequences) - set(seq_to_entry))
    if missing_mapping:
        LOGGER.warning("Missing manifest mapping for %d validation IDs; inferring from sequence_id", len(missing_mapping))
        for seq_id in missing_mapping:
            seq_to_entry[seq_id] = infer_entry_id_from_seq_id(seq_id)

    unique_entries = sorted(set(seq_to_entry[sid] for sid in val_sequences))
    LOGGER.info("Unique validation entries: %d", len(unique_entries))

    download_results: List[DownloadResult] = []
    total_bytes = 0
    if not args.skip_download:
        LOGGER.info("Downloading alignments to %s", openproteinnet_dir / args.subset)
        download_results, total_bytes = batch_download(
            entry_ids=unique_entries,
            openproteinnet_dir=openproteinnet_dir,
            subset=args.subset,
            base_url=args.base_url,
            timeout_sec=args.timeout_sec,
            workers=args.workers,
            force=args.force,
        )
    else:
        LOGGER.info("Skipping download (--skip-download)")

    verify = verify_alignments(
        val_sequences=val_sequences,
        seq_to_entry=seq_to_entry,
        openproteinnet_dir=openproteinnet_dir,
        subset=args.subset,
    )

    status_counts: Dict[str, int] = {}
    failed_downloads: List[Dict[str, str]] = []
    for res in download_results:
        status_counts[res.status] = status_counts.get(res.status, 0) + 1
        if res.status == "failed":
            failed_downloads.append({"entry_id": res.entry_id, "error": res.error or ""})

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "openproteinnet_dir": str(openproteinnet_dir),
        "subset": args.subset,
        "num_val_sequences": len(val_sequences),
        "num_unique_val_entries": len(unique_entries),
        "download": {
            "skipped": bool(args.skip_download),
            "workers": args.workers,
            "timeout_sec": args.timeout_sec,
            "force": bool(args.force),
            "total_downloaded_bytes": total_bytes,
            "status_counts": status_counts,
            "failed_examples": failed_downloads[:50],
        },
        "validation": verify,
    }

    summary_path = args.summary_path or (dataset_dir / "openproteinnet_val_alignment_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    LOGGER.info("Summary written to %s", summary_path)

    hard_fail = (
        verify["num_missing_alignment"] > 0
        or verify["num_parse_errors"] > 0
        or verify["num_header_mismatches"] > 0
        or verify["num_mismatches"] > 0
        or status_counts.get("failed", 0) > 0
    )
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
