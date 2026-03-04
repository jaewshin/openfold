#!/usr/bin/env python3
"""Run retrieval-based OpenFold on OpenProteinNet validation set.

This script mirrors the vanilla validation runner but generates alignment
directories via Pipeline A retrieval + embedding-scored SW alignment before
running OpenFold in precomputed-alignment mode.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from align.embed_align import build_alignment_result, gate_and_rank, smith_waterman_affine
from msa.a3m_writer import project_trace_to_a3m_row, validate_a3m_invariants, write_openfold_alignment_dir
from pipeline_a.embeddings import build_embedder, l2_normalize_rows
from pipeline_a.sequence_store import load_sequences_for_ids
from retrieve.retrieve_topk import IdTextLookup, FaissSearcher, _cheap_filter_and_dedup, _select_top_k
from scripts.retrieval.run_openproteinnet_val_vanilla_openfold import (
    evaluate_predictions,
    load_validation_records,
    run_openfold_inference,
    write_scores_csv,
)

LOGGER = logging.getLogger("run_openproteinnet_val_retrieval_openfold")


def _summ(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {"min": float(min(values)), "max": float(max(values)), "mean": float(mean(values))}


def _write_query_fasta(path: Path, target_id: str, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f">{target_id}\n{sequence}\n")


def _prepare_retrieval_alignments(
    *,
    records,
    output_root: Path,
    index_path: Path,
    ids_path: Path,
    ids_offsets_path: Path | None,
    sequence_fasta: Path | None,
    sequence_sqlite: Path | None,
    build_sequence_sqlite_if_missing: bool,
    retrieval_embedder_name: str,
    retrieval_device: str,
    alignment_embedder_name: str,
    alignment_device: str,
    normalize_query: bool,
    top_k: int,
    top_k_prime: int,
    length_ratio_low: float,
    length_ratio_high: float,
    scale: float,
    bias: float,
    gap_open: float,
    gap_extend: float,
    min_query_coverage: float,
    min_aligned_query_len: int,
    min_score_density: float,
    max_gap_frac: float,
    max_rows: int,
) -> Tuple[Path, Path, Dict[str, object]]:
    inputs_fasta_dir = output_root / "inputs" / "fasta"
    alignments_root = output_root / "inputs" / "alignments"
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Reuse heavy objects across all queries.
    retrieval_embedder = build_embedder(retrieval_embedder_name, device=retrieval_device)
    alignment_embedder = build_embedder(alignment_embedder_name, device=alignment_device)
    searcher = FaissSearcher(index_path=index_path)
    id_lookup = IdTextLookup(ids_path=ids_path, offsets_path=ids_offsets_path)

    LOGGER.info("Retrieval embedder=%s alignment embedder=%s", retrieval_embedder_name, alignment_embedder_name)
    LOGGER.info("Preparing retrieval alignments for %d validation targets...", len(records))

    gated_rows_counts: List[int] = []
    q_coverage_means: List[float] = []
    score_density_means: List[float] = []
    gap_frac_means: List[float] = []
    retrieval_seconds: List[float] = []
    alignment_seconds: List[float] = []
    a3m_num_sequences: List[int] = []
    target_reports: List[Dict[str, object]] = []

    for i, rec in enumerate(records, start=1):
        q_t0 = time.perf_counter()
        target_id = rec.sequence_id
        query_sequence = rec.sequence

        _write_query_fasta(inputs_fasta_dir / f"{target_id}.fasta", target_id, query_sequence)

        query_embedding = retrieval_embedder.embed_sequence(query_sequence).astype(np.float32)
        if normalize_query:
            query_embedding = l2_normalize_rows(query_embedding.reshape(1, -1))[0]

        top_rows = _select_top_k(
            searcher=searcher,
            id_lookup=id_lookup,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        ids_to_fetch = [seq_id for _, seq_id, _ in top_rows]
        records_map = load_sequences_for_ids(
            seq_ids=ids_to_fetch,
            fasta_path=sequence_fasta,
            sqlite_path=sequence_sqlite,
            build_sqlite_if_missing=build_sequence_sqlite_if_missing,
        )
        sequence_map: Mapping[str, str] = {sid: rec_.sequence for sid, rec_ in records_map.items()}
        candidates = _cheap_filter_and_dedup(
            query_length=len(query_sequence),
            rows=top_rows,
            sequence_map=sequence_map,
            length_ratio_low=length_ratio_low,
            length_ratio_high=length_ratio_high,
            top_k_prime=top_k_prime,
        )
        q_t1 = time.perf_counter()

        query_residues = alignment_embedder.embed_residues(query_sequence)
        aligned_rows = []
        for cand in candidates:
            cand_residues = alignment_embedder.embed_residues(cand.sequence)
            trace = smith_waterman_affine(
                query_residue_embeddings=query_residues,
                candidate_residue_embeddings=cand_residues,
                scale=scale,
                bias=bias,
                gap_open=gap_open,
                gap_extend=gap_extend,
            )
            aligned_rows.append(
                build_alignment_result(
                    seq_id=cand.seq_id,
                    ann_score=cand.ann_score,
                    sequence=cand.sequence,
                    trace=trace,
                    query_length=len(query_sequence),
                )
            )

        gated_rows = gate_and_rank(
            aligned_rows,
            min_query_coverage=min_query_coverage,
            min_aligned_query_len=min_aligned_query_len,
            min_score_density=min_score_density,
            max_gap_frac=max_gap_frac,
            max_rows=max_rows,
        )

        projected_rows = []
        for row in gated_rows:
            header = (
                f"retriever|{row.seq_id}|ann={row.ann_score:.4f}|sw={row.sw_score:.2f}|"
                f"cov={row.query_coverage:.3f}|gap={row.gap_frac:.3f}"
            )
            projected_rows.append(
                project_trace_to_a3m_row(
                    query_length=len(query_sequence),
                    candidate_sequence=row.sequence,
                    q_start=row.q_start,
                    s_start=row.s_start,
                    ops=row.ops,
                    seq_id=row.seq_id,
                    header=header,
                    drop_trailing_insertions=True,
                )
            )

        target_alignment_dir = write_openfold_alignment_dir(
            alignments_root=alignments_root,
            target_id=target_id,
            query_sequence=query_sequence,
            a3m_rows=projected_rows,
        )
        a3m_stats = validate_a3m_invariants(
            a3m_path=target_alignment_dir / "bfd_uniclust_hits.a3m",
            query_sequence=query_sequence,
        )
        q_t2 = time.perf_counter()

        retrieval_seconds.append(float(q_t1 - q_t0))
        alignment_seconds.append(float(q_t2 - q_t1))
        gated_rows_counts.append(len(gated_rows))
        a3m_num_sequences.append(int(a3m_stats["num_sequences"]))
        if gated_rows:
            q_coverage_means.append(float(np.mean([r.query_coverage for r in gated_rows])))
            score_density_means.append(float(np.mean([r.score_density for r in gated_rows])))
            gap_frac_means.append(float(np.mean([r.gap_frac for r in gated_rows])))
        else:
            q_coverage_means.append(0.0)
            score_density_means.append(0.0)
            gap_frac_means.append(1.0)

        target_reports.append(
            {
                "sequence_id": target_id,
                "entry_id": rec.entry_id,
                "query_length": len(query_sequence),
                "retrieved_rows": len(top_rows),
                "candidate_rows": len(candidates),
                "aligned_rows": len(aligned_rows),
                "gated_rows": len(gated_rows),
                "a3m_num_sequences": int(a3m_stats["num_sequences"]),
                "retrieval_seconds": float(q_t1 - q_t0),
                "alignment_seconds": float(q_t2 - q_t1),
            }
        )

        if i % 25 == 0 or i == len(records):
            LOGGER.info(
                "Prepared %d/%d targets (avg retrieval %.2fs, avg align %.2fs, avg gated %.1f)",
                i,
                len(records),
                float(np.mean(retrieval_seconds)) if retrieval_seconds else 0.0,
                float(np.mean(alignment_seconds)) if alignment_seconds else 0.0,
                float(np.mean(gated_rows_counts)) if gated_rows_counts else 0.0,
            )

    per_target_path = reports_dir / "retrieval_alignment_per_target.jsonl"
    with per_target_path.open("w", encoding="utf-8") as handle:
        for row in target_reports:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    prep_summary = {
        "num_records": len(records),
        "retrieval_embedder": retrieval_embedder_name,
        "alignment_embedder": alignment_embedder_name,
        "retrieval_device": retrieval_device,
        "alignment_device": alignment_device,
        "top_k": top_k,
        "top_k_prime": top_k_prime,
        "max_rows": max_rows,
        "row_quality": {
            "gated_rows_per_target": _summ(gated_rows_counts),
            "mean_query_coverage": _summ(q_coverage_means),
            "mean_score_density": _summ(score_density_means),
            "mean_gap_frac": _summ(gap_frac_means),
            "a3m_num_sequences": _summ(a3m_num_sequences),
        },
        "timings_sec": {
            "retrieval_per_target": _summ(retrieval_seconds),
            "alignment_per_target": _summ(alignment_seconds),
            "retrieval_total": float(np.sum(retrieval_seconds)) if retrieval_seconds else 0.0,
            "alignment_total": float(np.sum(alignment_seconds)) if alignment_seconds else 0.0,
        },
        "paths": {
            "fasta_dir": str(inputs_fasta_dir),
            "alignments_root": str(alignments_root),
            "per_target_jsonl": str(per_target_path),
        },
    }
    (reports_dir / "retrieval_prepare_summary.json").write_text(json.dumps(prep_summary, indent=2) + "\n")
    return inputs_fasta_dir, alignments_root, prep_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval-based OpenFold on OpenProteinNet validation set with shared scorer/output format."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet/retrieval_ready"),
    )
    parser.add_argument(
        "--openproteinnet-dir",
        type=Path,
        default=Path("/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet/retrieval_openfold_val"),
    )
    parser.add_argument("--subset", default="uniclust30")

    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("/insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/index/u50_esm2_35M.index"),
    )
    parser.add_argument(
        "--ids-path",
        type=Path,
        default=Path("/insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/index/u50_esm2_35M_ids.txt"),
    )
    parser.add_argument(
        "--ids-offsets-path",
        type=Path,
        default=Path("/insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/index/u50_esm2_35M_ids.txt.offsets.u64"),
    )
    parser.add_argument(
        "--sequence-sqlite",
        type=Path,
        default=Path("/insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/uniref50.seqio.sqlite"),
    )
    parser.add_argument(
        "--sequence-fasta",
        type=Path,
        default=Path("/insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/uniref50.fasta"),
    )
    parser.add_argument("--build-sequence-sqlite-if-missing", action="store_true")

    parser.add_argument("--retrieval-embedder", default="esm2_35m", choices=["esm2_35m", "aa_onehot"])
    parser.add_argument("--retrieval-device", default="cuda:0")
    parser.add_argument("--alignment-embedder", default="aa_onehot", choices=["esm2_35m", "aa_onehot"])
    parser.add_argument("--alignment-device", default="cpu")
    parser.add_argument("--normalize-query", action="store_true", default=True)
    parser.add_argument("--no-normalize-query", dest="normalize_query", action="store_false")

    # Pragmatic defaults for full-validation runtime.
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--top-k-prime", type=int, default=64)
    parser.add_argument("--length-ratio-low", type=float, default=0.7)
    parser.add_argument("--length-ratio-high", type=float, default=1.3)

    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--gap-open", type=float, default=-8.0)
    parser.add_argument("--gap-extend", type=float, default=-0.5)
    parser.add_argument("--min-query-coverage", type=float, default=0.15)
    parser.add_argument("--min-aligned-query-len", type=int, default=30)
    parser.add_argument("--min-score-density", type=float, default=1.0)
    parser.add_argument("--max-gap-frac", type=float, default=0.85)
    parser.add_argument("--max-rows", type=int, default=64)

    parser.add_argument("--template-mmcif-dir", type=Path, default=Path("tests/test_data/mmcifs"))
    parser.add_argument("--run-pretrained-path", type=Path, default=Path("run_pretrained_openfold.py"))
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("openfold/resources/openfold_params/finetuning_no_templ_ptm_1.pt"))
    parser.add_argument("--config-preset", default="finetuning_no_templ_ptm")
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--openfold-extra-args", nargs="*", default=[])

    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")

    output_root = args.output_root
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    records_all = load_validation_records(args.dataset_dir)
    records = records_all[args.offset :] if args.offset > 0 else records_all
    if args.max_targets > 0:
        records = records[: args.max_targets]
    if not records:
        raise RuntimeError("No validation records selected")

    prepare_summary: Dict[str, object] = {}
    fasta_dir = output_root / "inputs" / "fasta"
    alignments_root = output_root / "inputs" / "alignments"

    if not args.skip_prepare:
        fasta_dir, alignments_root, prepare_summary = _prepare_retrieval_alignments(
            records=records,
            output_root=output_root,
            index_path=args.index_path,
            ids_path=args.ids_path,
            ids_offsets_path=args.ids_offsets_path,
            sequence_fasta=args.sequence_fasta,
            sequence_sqlite=args.sequence_sqlite,
            build_sequence_sqlite_if_missing=args.build_sequence_sqlite_if_missing,
            retrieval_embedder_name=args.retrieval_embedder,
            retrieval_device=args.retrieval_device,
            alignment_embedder_name=args.alignment_embedder,
            alignment_device=args.alignment_device,
            normalize_query=args.normalize_query,
            top_k=args.top_k,
            top_k_prime=args.top_k_prime,
            length_ratio_low=args.length_ratio_low,
            length_ratio_high=args.length_ratio_high,
            scale=args.scale,
            bias=args.bias,
            gap_open=args.gap_open,
            gap_extend=args.gap_extend,
            min_query_coverage=args.min_query_coverage,
            min_aligned_query_len=args.min_aligned_query_len,
            min_score_density=args.min_score_density,
            max_gap_frac=args.max_gap_frac,
            max_rows=args.max_rows,
        )
    else:
        LOGGER.info("Skipping retrieval alignment preparation (--skip-prepare)")

    if args.prepare_only:
        LOGGER.info("Stopping after preparation (--prepare-only)")
        return 0

    run_dir = output_root / "run"
    predictions_dir = run_dir / "predictions"
    inference_return_code = 0
    if not args.skip_inference:
        python_exec = args.python_executable if args.python_executable else "/insomnia001/depts/pmg/users/js6118/miniforge3/envs/openfold_dev/bin/python"
        inference_return_code = run_openfold_inference(
            run_pretrained_path=args.run_pretrained_path,
            python_executable=python_exec,
            fasta_dir=fasta_dir,
            template_mmcif_dir=args.template_mmcif_dir,
            alignments_dir=alignments_root,
            output_dir=run_dir,
            config_preset=args.config_preset,
            checkpoint_path=args.checkpoint_path,
            model_device=args.model_device,
            extra_args=args.openfold_extra_args,
        )
        LOGGER.info("OpenFold inference finished with return code: %d", inference_return_code)
    else:
        LOGGER.info("Skipping inference (--skip-inference)")

    rows, eval_summary = evaluate_predictions(
        records=records,
        predictions_dir=predictions_dir,
        config_preset=args.config_preset,
    )
    write_scores_csv(rows, reports_dir / "per_target_scores.csv")

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(args.dataset_dir),
        "openproteinnet_dir": str(args.openproteinnet_dir),
        "output_root": str(output_root),
        "subset": args.subset,
        "config_preset": args.config_preset,
        "checkpoint_path": str(args.checkpoint_path),
        "model_device": args.model_device,
        "num_records_total": len(records_all),
        "offset": args.offset,
        "num_records_requested": len(records),
        "retrieval_settings": {
            "index_path": str(args.index_path),
            "ids_path": str(args.ids_path),
            "ids_offsets_path": str(args.ids_offsets_path) if args.ids_offsets_path else None,
            "sequence_sqlite": str(args.sequence_sqlite) if args.sequence_sqlite else None,
            "sequence_fasta": str(args.sequence_fasta) if args.sequence_fasta else None,
            "retrieval_embedder": args.retrieval_embedder,
            "retrieval_device": args.retrieval_device,
            "alignment_embedder": args.alignment_embedder,
            "alignment_device": args.alignment_device,
            "top_k": args.top_k,
            "top_k_prime": args.top_k_prime,
            "max_rows": args.max_rows,
            "length_ratio_low": args.length_ratio_low,
            "length_ratio_high": args.length_ratio_high,
            "scale": args.scale,
            "bias": args.bias,
            "gap_open": args.gap_open,
            "gap_extend": args.gap_extend,
            "min_query_coverage": args.min_query_coverage,
            "min_aligned_query_len": args.min_aligned_query_len,
            "min_score_density": args.min_score_density,
            "max_gap_frac": args.max_gap_frac,
        },
        "prepare": prepare_summary,
        "inference_return_code": inference_return_code,
        "evaluation": eval_summary,
    }

    summary_path = reports_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    LOGGER.info("Summary written to %s", summary_path)

    return int(inference_return_code)


if __name__ == "__main__":
    raise SystemExit(main())
