#!/usr/bin/env python3
"""Evaluate vanilla OpenFold SoloSeq on validation data (no retrieval).

This script uses retrieval-ready manifest records as a convenient validation
data source, but runs plain AlphaFold forward + AlphaFoldLoss only.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Tuple

# Allow running directly from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytorch_lightning as pl
import torch

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.np import residue_constants
from openfold.utils.import_weights import convert_deprecated_v1_keys
from openfold.utils.loss import AlphaFoldLoss, lddt_ca
from openfold.utils.superimposition import superimpose
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.validation_metrics import drmsd, gdt_ha, gdt_ts
from scripts.retrieval.retrieval_data import RetrievalDataModule

logger = logging.getLogger(__name__)


def _to_tensor(value, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    return torch.tensor(value, dtype=torch.float32, device=device)


def _extract_state_dict(
    checkpoint_obj: Mapping[str, object], prefer_ema: bool = True
) -> Tuple[Mapping[str, torch.Tensor], str]:
    if prefer_ema:
        ema = checkpoint_obj.get("ema")
        if isinstance(ema, Mapping):
            params = ema.get("params")
            if isinstance(params, Mapping):
                return params, "ema.params"

    for key in ("state_dict", "model_state_dict", "module"):
        value = checkpoint_obj.get(key)
        if isinstance(value, Mapping):
            return value, key

    if isinstance(checkpoint_obj, Mapping):
        return checkpoint_obj, "root"

    raise ValueError(f"Unsupported checkpoint object type: {type(checkpoint_obj).__name__}")


def _remap_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = (
        "module.model.",
        "model.openfold.",
        "module.openfold.",
        "model.",
        "module.",
        "openfold.",
    )
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = str(key)
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        remapped[new_key] = value
    return remapped


def _maybe_convert_legacy_keys(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Map older OpenFold checkpoint key names (e.g., `core.*`) to current names.
    return convert_deprecated_v1_keys(dict(state_dict))


def load_openfold_weights(
    model: AlphaFold,
    checkpoint_path: Path,
    *,
    prefer_ema: bool = True,
    strict: bool = False,
) -> Dict[str, object]:
    checkpoint_obj = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint_obj, Mapping):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} is not a dict-like object: {type(checkpoint_obj).__name__}"
        )

    raw_sd, source = _extract_state_dict(checkpoint_obj, prefer_ema=prefer_ema)
    remapped = _remap_state_dict_keys(raw_sd)
    converted = _maybe_convert_legacy_keys(remapped)
    missing, unexpected = model.load_state_dict(converted, strict=strict)

    return {
        "path": str(checkpoint_path),
        "source": source,
        "strict": strict,
        "loaded_keys": len(converted),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


def compute_validation_metrics(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    eps: float,
    *,
    compute_superimposition_metrics: bool = True,
) -> Dict[str, torch.Tensor]:
    metrics: Dict[str, torch.Tensor] = {}

    gt_coords = batch["all_atom_positions"]
    pred_coords = outputs["final_atom_positions"]
    all_atom_mask = batch["all_atom_mask"]

    gt_coords_masked = gt_coords * all_atom_mask[..., None]
    pred_coords_masked = pred_coords * all_atom_mask[..., None]
    ca_pos = residue_constants.atom_order["CA"]
    gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
    pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
    all_atom_mask_ca = all_atom_mask[..., ca_pos]

    metrics["lddt_ca"] = lddt_ca(
        pred_coords,
        gt_coords,
        all_atom_mask,
        eps=eps,
        per_residue=False,
    )

    metrics["drmsd_ca"] = drmsd(
        pred_coords_masked_ca,
        gt_coords_masked_ca,
        mask=all_atom_mask_ca,
    )

    if compute_superimposition_metrics:
        # Some fixtures can contain invalid structures (e.g. all-NaN coordinates)
        # that produce zero usable CA atoms. Compute superimposition metrics only
        # on valid samples and emit NaN for skipped ones.
        batch_dims = all_atom_mask_ca.shape[:-1]
        flat_ref = gt_coords_masked_ca.reshape(-1, gt_coords_masked_ca.shape[-2], gt_coords_masked_ca.shape[-1])
        flat_pred = pred_coords_masked_ca.reshape(
            -1, pred_coords_masked_ca.shape[-2], pred_coords_masked_ca.shape[-1]
        )
        flat_mask = all_atom_mask_ca.reshape(-1, all_atom_mask_ca.shape[-1])

        flat_super = flat_pred.clone()
        flat_alignment_rmsd = torch.full(
            (flat_mask.shape[0],),
            torch.nan,
            device=flat_pred.device,
            dtype=flat_pred.dtype,
        )
        flat_gdt_ts = torch.full_like(flat_alignment_rmsd, torch.nan)
        flat_gdt_ha = torch.full_like(flat_alignment_rmsd, torch.nan)

        for i in range(flat_mask.shape[0]):
            if float(flat_mask[i].sum().item()) <= 0.0:
                continue

            super_i, rmsd_i = superimpose(
                flat_ref[i : i + 1],
                flat_pred[i : i + 1],
                flat_mask[i : i + 1],
            )
            flat_super[i] = super_i[0]
            flat_alignment_rmsd[i] = rmsd_i.reshape(-1)[0]
            flat_gdt_ts[i] = gdt_ts(
                super_i,
                flat_ref[i : i + 1],
                flat_mask[i : i + 1],
            ).reshape(-1)[0]
            flat_gdt_ha[i] = gdt_ha(
                super_i,
                flat_ref[i : i + 1],
                flat_mask[i : i + 1],
            ).reshape(-1)[0]

        metrics["alignment_rmsd"] = flat_alignment_rmsd.reshape(batch_dims)
        metrics["gdt_ts"] = flat_gdt_ts.reshape(batch_dims)
        metrics["gdt_ha"] = flat_gdt_ha.reshape(batch_dims)

    return metrics


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate vanilla OpenFold SoloSeq on validation split (no retrieval)."
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=Path("experiments/data/retrieval_ready_mmseqs/manifest.jsonl"),
        help="Path to retrieval-ready manifest JSONL containing train/val split rows.",
    )
    parser.add_argument(
        "--seq_embedding_dir",
        type=Path,
        default=Path("experiments/data/retrieval_ready_mmseqs/seq_embedding_esm1b"),
        help="Directory containing per-sequence ESM1b embeddings (.npy/.pt).",
    )
    parser.add_argument(
        "--openfold_checkpoint",
        type=Path,
        default=Path("experiments/data/openfold_soloseq_params/seq_model_esm1b_ptm.pt"),
        help="OpenFold SoloSeq checkpoint path.",
    )
    parser.add_argument(
        "--config_preset",
        type=str,
        default="seq_model_esm1b_ptm",
        help="OpenFold config preset to instantiate the model.",
    )
    parser.add_argument(
        "--data_config_preset",
        type=str,
        default="seqemb_initial_training",
        help=(
            "Config preset used by the data feature pipeline. "
            "For SoloSeq validation this should keep eval max_msa_clusters=1."
        ),
    )
    parser.add_argument(
        "--disable_templates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable template stack/features (recommended for retrieval-ready fixtures).",
    )
    parser.add_argument(
        "--strict_checkpoint_loading",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, fail when checkpoint/model keys do not match exactly.",
    )
    parser.add_argument(
        "--prefer_ema_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer EMA weights when available in the checkpoint.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seq_embedding_dim", type=int, default=1280)
    parser.add_argument("--max_recycling_iters", type=int, default=0)
    parser.add_argument(
        "--strict_seq_embeddings",
        action="store_true",
        help="Fail if any sequence embedding file is missing.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device.",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="Autocast precision (CUDA only).",
    )
    parser.add_argument(
        "--compute_superimposition_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute alignment_rmsd, gdt_ts, and gdt_ha.",
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=0,
        help="If >0, evaluate only the first N validation batches (smoke mode).",
    )
    parser.add_argument(
        "--log_every_n_batches",
        type=int,
        default=50,
        help="Progress logging interval in batches.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to logs/soloseq_baseline_val_metrics_<timestamp>.json",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    pl.seed_everything(args.seed, workers=True)

    if args.batch_size < 1:
        raise SystemExit("--batch_size must be >= 1")
    if args.num_workers < 0:
        raise SystemExit("--num_workers must be >= 0")
    if args.max_recycling_iters < 0:
        raise SystemExit("--max_recycling_iters must be >= 0")
    if args.max_val_batches < 0:
        raise SystemExit("--max_val_batches must be >= 0")

    manifest_path = args.manifest_path.expanduser().resolve()
    seq_embedding_dir = args.seq_embedding_dir.expanduser().resolve()
    checkpoint_path = args.openfold_checkpoint.expanduser().resolve()

    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    if not seq_embedding_dir.is_dir():
        raise SystemExit(f"Embedding dir not found: {seq_embedding_dir}")
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    device = _resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but no CUDA device is available.")

    logger.info("Using device=%s precision=%s", device, args.precision)
    logger.info("Manifest: %s", manifest_path)
    logger.info("Seq embeddings: %s", seq_embedding_dir)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info(
        "Model config preset: %s | Data config preset: %s",
        args.config_preset,
        args.data_config_preset,
    )

    data_module = RetrievalDataModule(
        manifest_path=manifest_path,
        seq_embedding_dir=seq_embedding_dir,
        config_preset=args.data_config_preset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        disable_templates=args.disable_templates,
        max_recycling_iters=args.max_recycling_iters,
        strict_seq_embeddings=args.strict_seq_embeddings,
        seq_embedding_dim=args.seq_embedding_dim,
    )
    data_module.setup(stage="validate")
    val_loader = data_module.val_dataloader()
    logger.info("Validation batches: %d", len(val_loader))

    config = model_config(args.config_preset, train=False, low_prec=False)
    if args.disable_templates:
        config.model.template.enabled = False
        config.data.common.use_templates = False

    model = AlphaFold(config).to(device)
    loss_fn = AlphaFoldLoss(config.loss).to(device)

    load_info = load_openfold_weights(
        model,
        checkpoint_path=checkpoint_path,
        prefer_ema=args.prefer_ema_weights,
        strict=args.strict_checkpoint_loading,
    )
    logger.info(
        "Loaded checkpoint (%s): source=%s loaded_keys=%d missing=%d unexpected=%d strict=%s",
        load_info["path"],
        load_info["source"],
        load_info["loaded_keys"],
        load_info["missing_keys"],
        load_info["unexpected_keys"],
        load_info["strict"],
    )

    model.eval()
    loss_fn.eval()

    aggregate_sum = defaultdict(float)
    aggregate_count = defaultdict(int)
    num_batches = 0
    num_samples = 0
    skipped_empty_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                break

            if batch is None:
                skipped_empty_batches += 1
                continue

            tensor_batch = {
                k: v.to(device=device, non_blocking=True)
                for k, v in batch.items()
                if torch.is_tensor(v)
            }
            if not tensor_batch:
                continue

            with _autocast_context(device, args.precision):
                outputs = model(tensor_batch)
                labels = tensor_tree_map(lambda t: t[..., -1], tensor_batch)
                labels["use_clamped_fape"] = torch.zeros_like(labels["use_clamped_fape"])
                loss, breakdown = loss_fn(outputs, labels, _return_breakdown=True)
                metrics = compute_validation_metrics(
                    labels,
                    outputs,
                    eps=config.globals.eps,
                    compute_superimposition_metrics=args.compute_superimposition_metrics,
                )

            bsz = int(labels["aatype"].shape[0])
            num_samples += bsz
            num_batches += 1

            all_scalars = {"loss": loss}
            all_scalars.update({f"loss/{k}": v for k, v in breakdown.items()})
            all_scalars.update({f"metric/{k}": v for k, v in metrics.items()})

            for name, value in all_scalars.items():
                t = _to_tensor(value, device=device).detach().float().reshape(-1)
                finite = torch.isfinite(t)
                if not finite.any():
                    continue
                aggregate_sum[name] += float(t[finite].sum().item())
                aggregate_count[name] += int(finite.sum().item())

            if args.log_every_n_batches > 0 and (
                (batch_idx + 1) % args.log_every_n_batches == 0 or batch_idx == 0
            ):
                running_loss = math.nan
                if aggregate_count["loss"] > 0:
                    running_loss = aggregate_sum["loss"] / aggregate_count["loss"]
                logger.info(
                    "Processed batch %d (samples=%d, running_loss=%.6f)",
                    batch_idx + 1,
                    num_samples,
                    running_loss,
                )

    if num_batches == 0:
        raise SystemExit("No validation batches were processed.")

    if skipped_empty_batches > 0:
        logger.warning("Skipped %d empty validation batch(es) after filtering invalid samples.", skipped_empty_batches)

    means = {}
    for name, value_sum in aggregate_sum.items():
        count = aggregate_count[name]
        if count > 0:
            means[name] = value_sum / float(count)

    loss_breakdown = {
        k[len("loss/") :]: v for k, v in means.items() if k.startswith("loss/")
    }
    metric_breakdown = {
        k[len("metric/") :]: v for k, v in means.items() if k.startswith("metric/")
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_json is None:
        output_json = (REPO_ROOT / "logs" / f"soloseq_baseline_val_metrics_{timestamp}.json").resolve()
    else:
        output_json = args.output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": timestamp,
        "manifest_path": str(manifest_path),
        "seq_embedding_dir": str(seq_embedding_dir),
        "checkpoint": load_info,
        "model_config_preset": args.config_preset,
        "data_config_preset": args.data_config_preset,
        "disable_templates": args.disable_templates,
        "device": str(device),
        "precision": args.precision,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_recycling_iters": args.max_recycling_iters,
        "num_val_batches": num_batches,
        "num_val_samples": num_samples,
        "num_skipped_empty_batches": skipped_empty_batches,
        "loss": means.get("loss", math.nan),
        "loss_breakdown": dict(sorted(loss_breakdown.items())),
        "metrics": dict(sorted(metric_breakdown.items())),
    }
    output_json.write_text(json.dumps(summary, indent=2))

    print(f"Wrote evaluation summary to: {output_json}")
    print(f"val/loss: {summary['loss']:.6f}")
    for k, v in sorted(summary["metrics"].items()):
        print(f"val/{k}: {v:.6f}")


if __name__ == "__main__":
    main()
