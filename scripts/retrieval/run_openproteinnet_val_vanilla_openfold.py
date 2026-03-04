#!/usr/bin/env python3
"""Run vanilla OpenFold on OpenProteinNet validation set with precomputed MSAs.

Pipeline:
1) Prepare one FASTA per validation sequence.
2) Prepare one alignment subdirectory per sequence ID expected by run_pretrained_openfold.py.
3) Run OpenFold inference (no retrieval, only precomputed alignments).
4) Evaluate predictions against validation target structures.

Notes:
- This script is designed for monomer validation sets (sequence IDs like "<entry>_<chain>").
- It does not require template hits in alignment directories. We use a no-template preset by default.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from Bio import Align
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1


LOGGER = logging.getLogger("run_openproteinnet_val_vanilla_openfold")


def parse_fasta(path: Path) -> Dict[str, str]:
    records: Dict[str, List[str]] = {}
    current: Optional[str] = None
    with path.open("r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                records.setdefault(current, [])
            else:
                if current is None:
                    raise ValueError(f"Invalid FASTA {path}: sequence before header")
                records[current].append(line)
    return {k: "".join(v).upper() for k, v in records.items()}


@dataclass(frozen=True)
class ValRecord:
    sequence_id: str
    sequence: str
    entry_id: str
    chain_id: str
    structure_path: Path


def infer_entry_id(sequence_id: str) -> str:
    return sequence_id.split("_", 1)[0].upper()


def infer_chain_id(sequence_id: str) -> str:
    parts = sequence_id.split("_", 1)
    if len(parts) == 2 and parts[1]:
        return parts[1]
    return "A"


def load_validation_records(dataset_dir: Path) -> List[ValRecord]:
    val_fasta = dataset_dir / "val.fasta"
    manifest = dataset_dir / "manifest.jsonl"
    if not val_fasta.exists():
        raise FileNotFoundError(f"Missing validation FASTA: {val_fasta}")
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    val_map = parse_fasta(val_fasta)
    manifest_map: Dict[str, Dict[str, str]] = {}
    with manifest.open("r") as fh:
        for raw in fh:
            row = json.loads(raw)
            if str(row.get("split", "")) != "val":
                continue
            sid = str(row.get("sequence_id", "")).strip()
            if sid:
                manifest_map[sid] = row

    records: List[ValRecord] = []
    for sid, seq in val_map.items():
        row = manifest_map.get(sid, {})
        entry = str(row.get("domain_id", "")).strip().upper()
        if not entry:
            structure_path = str(row.get("structure_path", "")).strip()
            if structure_path:
                entry = Path(structure_path).parent.parent.name.upper()
        if not entry:
            entry = infer_entry_id(sid)

        chain = str(row.get("chain_id", "")).strip() or infer_chain_id(sid)
        struct_path = Path(str(row.get("structure_path", "")).strip()) if row.get("structure_path") else Path()
        if not struct_path.exists():
            # Fall back to canonical OpenProteinNet layout.
            struct_path = dataset_dir.parent / "uniclust30" / entry / "pdb" / f"{entry}.pdb"

        records.append(
            ValRecord(
                sequence_id=sid,
                sequence=seq,
                entry_id=entry,
                chain_id=chain,
                structure_path=struct_path,
            )
        )

    records.sort(key=lambda r: r.sequence_id)
    return records


def resolve_source_a3m(openproteinnet_dir: Path, subset: str, entry_id: str) -> Path:
    a3m = openproteinnet_dir / subset / entry_id / "a3m" / "uniclust30.a3m"
    if a3m.exists() and a3m.stat().st_size > 0:
        return a3m
    a3m_gz = openproteinnet_dir / subset / entry_id / "a3m" / "uniclust30.a3m.gz"
    if a3m_gz.exists() and a3m_gz.stat().st_size > 0:
        return a3m_gz
    raise FileNotFoundError(f"Alignment not found for {entry_id}: {a3m}(.gz)")


def ensure_fasta(path: Path, sequence_id: str, sequence: str, overwrite: bool) -> None:
    text = f">{sequence_id}\n{sequence}\n"
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def ensure_alignment_link_or_copy(src: Path, dst: Path, force_copy: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and not force_copy:
            target = os.readlink(dst)
            if target == str(src):
                return "exists"
        dst.unlink()

    if src.suffix == ".gz":
        with gzip.open(src, "rt") as in_fh, dst.open("w") as out_fh:
            shutil.copyfileobj(in_fh, out_fh)
        return "decompressed"

    if force_copy:
        shutil.copy2(src, dst)
        return "copied"

    os.symlink(src, dst)
    return "symlinked"


def prepare_inputs(
    *,
    records: Sequence[ValRecord],
    openproteinnet_dir: Path,
    subset: str,
    fasta_dir: Path,
    alignments_dir: Path,
    overwrite_fasta: bool,
    force_alignment_copy: bool,
) -> Dict[str, object]:
    stats = {
        "num_records": len(records),
        "fasta_written": 0,
        "alignment_symlinked": 0,
        "alignment_copied": 0,
        "alignment_decompressed": 0,
        "alignment_exists": 0,
        "missing_alignment": 0,
        "missing_structure": 0,
        "missing_alignment_examples": [],
        "missing_structure_examples": [],
    }

    for rec in records:
        fasta_path = fasta_dir / f"{rec.sequence_id}.fasta"
        before = fasta_path.exists()
        ensure_fasta(fasta_path, rec.sequence_id, rec.sequence, overwrite=overwrite_fasta)
        after = fasta_path.exists()
        if (not before) and after:
            stats["fasta_written"] += 1

        if not rec.structure_path.exists():
            stats["missing_structure"] += 1
            if len(stats["missing_structure_examples"]) < 20:
                stats["missing_structure_examples"].append(
                    {"sequence_id": rec.sequence_id, "structure_path": str(rec.structure_path)}
                )

        try:
            src = resolve_source_a3m(openproteinnet_dir, subset, rec.entry_id)
        except FileNotFoundError:
            stats["missing_alignment"] += 1
            if len(stats["missing_alignment_examples"]) < 20:
                stats["missing_alignment_examples"].append(
                    {"sequence_id": rec.sequence_id, "entry_id": rec.entry_id}
                )
            continue

        dst = alignments_dir / rec.sequence_id / "uniclust30.a3m"
        status = ensure_alignment_link_or_copy(src, dst, force_copy=force_alignment_copy)
        if status == "symlinked":
            stats["alignment_symlinked"] += 1
        elif status == "copied":
            stats["alignment_copied"] += 1
        elif status == "decompressed":
            stats["alignment_decompressed"] += 1
        elif status == "exists":
            stats["alignment_exists"] += 1

    return stats


def run_openfold_inference(
    *,
    run_pretrained_path: Path,
    python_executable: str,
    fasta_dir: Path,
    template_mmcif_dir: Path,
    alignments_dir: Path,
    output_dir: Path,
    config_preset: str,
    checkpoint_path: Path,
    model_device: str,
    max_recycling_iters: int,
    extra_args: Sequence[str],
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    forwarded_args = list(extra_args)

    # Preserve user-provided experiment config overrides and enforce recycling.
    merged_experiment_cfg: Dict[str, object] = {}
    while "--experiment_config_json" in forwarded_args:
        idx = forwarded_args.index("--experiment_config_json")
        if idx + 1 >= len(forwarded_args):
            raise ValueError("--experiment_config_json provided without a path in --extra-run-args")
        existing_cfg_path = Path(forwarded_args[idx + 1])
        if not existing_cfg_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {existing_cfg_path}")
        with existing_cfg_path.open("r") as fh:
            loaded = json.load(fh)
            if not isinstance(loaded, dict):
                raise ValueError(f"Experiment config must be a JSON object: {existing_cfg_path}")
            merged_experiment_cfg.update(loaded)
        del forwarded_args[idx : idx + 2]

    merged_experiment_cfg["data.common.max_recycling_iters"] = int(max_recycling_iters)
    recycling_cfg_path = output_dir / f"recycling_override_{os.getpid()}.json"
    recycling_cfg_path.write_text(json.dumps(merged_experiment_cfg, indent=2) + "\n")

    cmd = [
        python_executable,
        str(run_pretrained_path),
        str(fasta_dir),
        str(template_mmcif_dir),
        "--use_precomputed_alignments",
        str(alignments_dir),
        "--output_dir",
        str(output_dir),
        "--config_preset",
        config_preset,
        "--openfold_checkpoint_path",
        str(checkpoint_path),
        "--model_device",
        model_device,
        "--skip_relaxation",
    ]
    cmd.extend(forwarded_args)
    cmd.extend(["--experiment_config_json", str(recycling_cfg_path)])

    LOGGER.info("Running OpenFold inference on %d FASTA files", len(list(fasta_dir.glob("*.fasta"))))
    LOGGER.info("Command: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=False)
        return int(proc.returncode)
    finally:
        recycling_cfg_path.unlink(missing_ok=True)


THREE_TO_ONE = {k.upper(): v for k, v in protein_letters_3to1.items()}


def aa3_to_aa1(resname: str) -> str:
    return THREE_TO_ONE.get(resname.upper(), "X")


def extract_chain_ca(pdb_path: Path, preferred_chain: Optional[str]) -> Tuple[str, np.ndarray, np.ndarray]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    model = next(structure.get_models())

    chain = None
    if preferred_chain and preferred_chain in model:
        chain = model[preferred_chain]
    else:
        chain = next(model.get_chains())

    seq_chars: List[str] = []
    coords: List[np.ndarray] = []
    bfac: List[float] = []
    for residue in chain:
        if residue.id[0] != " ":
            continue
        if "CA" not in residue:
            continue
        atom = residue["CA"]
        seq_chars.append(aa3_to_aa1(residue.resname))
        coords.append(np.asarray(atom.coord, dtype=np.float64))
        bfac.append(float(atom.bfactor))

    if not coords:
        raise ValueError(f"No CA atoms found in {pdb_path} chain {chain.id}")

    return "".join(seq_chars), np.vstack(coords), np.asarray(bfac, dtype=np.float64)


def aligned_index_pairs(seq_a: str, seq_b: str) -> List[Tuple[int, int]]:
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(seq_a, seq_b)[0]

    a_blocks, b_blocks = alignment.aligned
    pairs: List[Tuple[int, int]] = []
    for (a_s, a_e), (b_s, b_e) in zip(a_blocks, b_blocks):
        block_len = min(a_e - a_s, b_e - b_s)
        for i in range(block_len):
            pairs.append((a_s + i, b_s + i))
    return pairs


def kabsch_align_rmsd(pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    pred_center = pred.mean(axis=0)
    target_center = target.mean(axis=0)
    p0 = pred - pred_center
    t0 = target - target_center
    cov = p0.T @ t0
    u, _, vt = np.linalg.svd(cov)
    v = vt.T
    ut = u.T
    d = np.sign(np.linalg.det(v @ ut))
    corr = np.eye(3)
    corr[2, 2] = d
    rot = v @ corr @ ut
    aligned = p0 @ rot
    diff = aligned - t0
    rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    return aligned, rmsd


def gdt(distances: np.ndarray, cutoffs: Sequence[float]) -> float:
    vals = [(distances <= c).mean() for c in cutoffs]
    return float(np.mean(vals))


def predict_file_for_sequence(predictions_dir: Path, sequence_id: str, config_preset: str) -> Optional[Path]:
    exact = predictions_dir / f"{sequence_id}_{config_preset}_unrelaxed.pdb"
    if exact.exists():
        return exact
    candidates = sorted(predictions_dir.glob(f"{sequence_id}_{config_preset}*_unrelaxed.pdb"))
    if candidates:
        return candidates[0]
    return None


def evaluate_predictions(
    *,
    records: Sequence[ValRecord],
    predictions_dir: Path,
    config_preset: str,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    missing_pred: List[str] = []
    failed_eval: List[Dict[str, str]] = []

    for rec in records:
        pred_path = predict_file_for_sequence(predictions_dir, rec.sequence_id, config_preset)
        if pred_path is None:
            missing_pred.append(rec.sequence_id)
            continue
        if not rec.structure_path.exists():
            failed_eval.append(
                {
                    "sequence_id": rec.sequence_id,
                    "error": f"missing_structure:{rec.structure_path}",
                }
            )
            continue

        try:
            pred_seq, pred_ca, pred_b = extract_chain_ca(pred_path, preferred_chain=None)
            true_seq, true_ca, _ = extract_chain_ca(rec.structure_path, preferred_chain=rec.chain_id)
            pairs = aligned_index_pairs(pred_seq, true_seq)
            if len(pairs) < 20:
                raise ValueError(f"Too few aligned residues: {len(pairs)}")

            pred_idx = np.asarray([i for i, _ in pairs], dtype=np.int64)
            true_idx = np.asarray([j for _, j in pairs], dtype=np.int64)
            p = pred_ca[pred_idx]
            t = true_ca[true_idx]
            p_aligned, ca_rmsd = kabsch_align_rmsd(p, t)
            dists = np.linalg.norm(p_aligned - (t - t.mean(axis=0)), axis=1)
            gdt_ts = gdt(dists, cutoffs=(1.0, 2.0, 4.0, 8.0))
            gdt_ha = gdt(dists, cutoffs=(0.5, 1.0, 2.0, 4.0))
            mean_plddt = float(np.mean(pred_b[pred_idx]))
            rows.append(
                {
                    "sequence_id": rec.sequence_id,
                    "entry_id": rec.entry_id,
                    "chain_id": rec.chain_id,
                    "pred_path": str(pred_path),
                    "target_path": str(rec.structure_path),
                    "n_aligned_ca": int(len(pairs)),
                    "pred_len_ca": int(len(pred_ca)),
                    "target_len_ca": int(len(true_ca)),
                    "mean_plddt": mean_plddt,
                    "ca_rmsd": float(ca_rmsd),
                    "gdt_ts": float(gdt_ts),
                    "gdt_ha": float(gdt_ha),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed_eval.append({"sequence_id": rec.sequence_id, "error": str(exc)})

    def summarize_metric(name: str) -> Dict[str, float]:
        vals = np.asarray([float(r[name]) for r in rows], dtype=np.float64)
        if vals.size == 0:
            return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
        }

    summary = {
        "num_targets": len(records),
        "num_scored": len(rows),
        "num_missing_predictions": len(missing_pred),
        "num_failed_evaluation": len(failed_eval),
        "missing_prediction_examples": missing_pred[:25],
        "failed_evaluation_examples": failed_eval[:25],
        "metrics": {
            "mean_plddt": summarize_metric("mean_plddt"),
            "ca_rmsd": summarize_metric("ca_rmsd"),
            "gdt_ts": summarize_metric("gdt_ts"),
            "gdt_ha": summarize_metric("gdt_ha"),
        },
    }
    return rows, summary


def write_scores_csv(rows: Iterable[Dict[str, object]], csv_path: Path) -> None:
    rows = list(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run + evaluate vanilla OpenFold on OpenProteinNet validation set with precomputed alignments."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet/retrieval_ready"),
        help="Directory containing val.fasta and manifest.jsonl.",
    )
    parser.add_argument(
        "--openproteinnet-dir",
        type=Path,
        default=Path("/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet"),
        help="OpenProteinNet root containing uniclust30/<entry>/a3m and pdb.",
    )
    parser.add_argument("--subset", default="uniclust30", help="OpenProteinNet subset.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet/vanilla_openfold_val"),
        help="Output root for prepared inputs, predictions, and reports.",
    )
    parser.add_argument(
        "--template-mmcif-dir",
        type=Path,
        default=Path("tests/test_data/mmcifs"),
        help="mmCIF dir required by run_pretrained_openfold.py (any dir with .cif files).",
    )
    parser.add_argument("--run-pretrained-path", type=Path, default=Path("run_pretrained_openfold.py"))
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("openfold/resources/openfold_params/finetuning_no_templ_ptm_1.pt"),
    )
    parser.add_argument("--config-preset", default="finetuning_no_templ_ptm")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite-fasta", action="store_true")
    parser.add_argument("--force-alignment-copy", action="store_true")
    parser.add_argument(
        "--max-targets",
        type=int,
        default=0,
        help="If >0, run only first N validation records (for smoke tests).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N validation records before selecting targets.",
    )
    parser.add_argument(
        "--extra-run-args",
        nargs="*",
        default=[],
        help="Additional args forwarded to run_pretrained_openfold.py",
    )
    parser.add_argument(
        "--max-recycling-iters",
        type=int,
        default=3,
        help="Override OpenFold recycling count for this vanilla validation run.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")

    records_all = load_validation_records(args.dataset_dir)
    records = records_all
    if args.offset > 0:
        records = records[args.offset :]
    if args.max_targets > 0:
        records = records[: args.max_targets]
    if not records:
        raise RuntimeError("No validation records found")

    output_root = args.output_root
    fasta_dir = output_root / "inputs" / "fasta"
    alignments_dir = output_root / "inputs" / "alignments"
    run_dir = output_root / "run"
    predictions_dir = run_dir / "predictions"
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    prepare_stats: Dict[str, object] = {}
    if not args.skip_prepare:
        LOGGER.info("Preparing OpenFold inputs for %d validation sequences...", len(records))
        prepare_stats = prepare_inputs(
            records=records,
            openproteinnet_dir=args.openproteinnet_dir,
            subset=args.subset,
            fasta_dir=fasta_dir,
            alignments_dir=alignments_dir,
            overwrite_fasta=args.overwrite_fasta,
            force_alignment_copy=args.force_alignment_copy,
        )
        (reports_dir / "prepare_summary.json").write_text(json.dumps(prepare_stats, indent=2) + "\n")
        LOGGER.info("Preparation complete: %s", json.dumps(prepare_stats, indent=2))
    else:
        LOGGER.info("Skipping preparation (--skip-prepare)")

    if args.prepare_only:
        LOGGER.info("Stopping after preparation (--prepare-only)")
        return 0

    inference_return_code = 0
    if not args.skip_inference:
        if not args.template_mmcif_dir.exists():
            raise FileNotFoundError(f"template_mmcif_dir not found: {args.template_mmcif_dir}")
        if not any(args.template_mmcif_dir.glob("*.cif")):
            raise RuntimeError(f"template_mmcif_dir has no .cif files: {args.template_mmcif_dir}")
        if not args.run_pretrained_path.exists():
            raise FileNotFoundError(f"run_pretrained_openfold.py not found: {args.run_pretrained_path}")
        if not args.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

        inference_return_code = run_openfold_inference(
            run_pretrained_path=args.run_pretrained_path,
            python_executable=args.python_executable,
            fasta_dir=fasta_dir,
            template_mmcif_dir=args.template_mmcif_dir,
            alignments_dir=alignments_dir,
            output_dir=run_dir,
            config_preset=args.config_preset,
            checkpoint_path=args.checkpoint_path,
            model_device=args.model_device,
            max_recycling_iters=args.max_recycling_iters,
            extra_args=args.extra_run_args,
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

    final_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(args.dataset_dir),
        "openproteinnet_dir": str(args.openproteinnet_dir),
        "output_root": str(output_root),
        "subset": args.subset,
        "config_preset": args.config_preset,
        "checkpoint_path": str(args.checkpoint_path),
        "model_device": args.model_device,
        "max_recycling_iters": args.max_recycling_iters,
        "num_records_total": len(records_all),
        "offset": args.offset,
        "num_records_requested": len(records),
        "prepare": prepare_stats,
        "inference_return_code": inference_return_code,
        "evaluation": eval_summary,
    }
    (reports_dir / "summary.json").write_text(json.dumps(final_summary, indent=2) + "\n")
    print(json.dumps(final_summary, indent=2))

    if inference_return_code != 0:
        return inference_return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
