#!/usr/bin/env python3
"""YAML-driven modular retrieval-augmented OpenFold training entrypoint."""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

import pytorch_lightning as pl
import torch

# Allow running directly from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openfold.model.retrieval.lightning_module import RetrievalAugmentedLightningModule
    from scripts.retrieval.retrieval_data import RetrievalDataModule


def _deep_merge(base: Dict, update: Dict) -> Dict:
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_config(path: Path, _visited: Optional[set] = None) -> Dict:
    try:
        import yaml
    except Exception as exc:
        raise SystemExit("PyYAML is required for --config parsing") from exc

    path = path.resolve()
    _visited = set() if _visited is None else _visited
    if path in _visited:
        raise ValueError(f"Cyclic config defaults reference detected at: {path}")
    _visited.add(path)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")

    defaults = data.pop("defaults", [])
    if defaults is None:
        defaults = []
    if not isinstance(defaults, list):
        raise ValueError(f"'defaults' must be a list in config: {path}")

    merged: Dict = {}
    for rel in defaults:
        child_path = (path.parent / str(rel)).resolve()
        child_cfg = _load_config(child_path, _visited=_visited)
        merged = _deep_merge(merged, child_cfg)

    merged = _deep_merge(merged, data)
    return merged


def _set_nested(cfg: Dict, dotted_key: str, value):
    parts = dotted_key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _apply_overrides(cfg: Dict, overrides: Sequence[str]) -> Dict:
    try:
        import yaml
    except Exception as exc:
        raise SystemExit("PyYAML is required for --set overrides") from exc

    out = dict(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set override (expected key=value): {item}")
        key, raw_val = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set override key: {item}")
        value = yaml.safe_load(raw_val)
        _set_nested(out, key, value)
    return out


def _auto_accelerator() -> str:
    return "gpu" if torch.cuda.is_available() else "cpu"


def _normalize_checkpoint_path(path_like: str, base_dir: Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _resolve_openfold_checkpoint(model_cfg: Dict, base_dir: Path) -> Optional[str]:
    raw_checkpoint = model_cfg.get("openfold_checkpoint", None)
    if raw_checkpoint is None:
        return None

    raw_str = str(raw_checkpoint).strip()
    if raw_str == "" or raw_str.lower() == "null":
        return None

    if raw_str.lower() != "auto":
        checkpoint_path = _normalize_checkpoint_path(raw_str, base_dir=base_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Configured model.openfold_checkpoint does not exist: {checkpoint_path}")
        return str(checkpoint_path)

    search_globs = model_cfg.get(
        "openfold_checkpoint_search_globs",
        [
            "outputs/**/*.ckpt",
            "checkpoints/**/*.ckpt",
            "logs/**/*.ckpt",
        ],
    )
    if not isinstance(search_globs, list) or not search_globs:
        raise ValueError(
            "model.openfold_checkpoint=auto requires a non-empty list at model.openfold_checkpoint_search_globs"
        )

    candidates: List[Path] = []
    for pattern in search_globs:
        abs_pattern = str(_normalize_checkpoint_path(str(pattern), base_dir=base_dir))
        for match in glob.glob(abs_pattern, recursive=True):
            p = Path(match)
            if p.is_file():
                candidates.append(p)

    if not candidates:
        return None

    best = max(candidates, key=lambda p: p.stat().st_mtime)
    logger.info("Auto-resolved OpenFold checkpoint: %s", best)
    return str(best)


def _validate_model_cfg(cfg: Dict, base_dir: Path) -> Dict:
    out = dict(cfg)
    model_cfg = dict(out.get("model", {}))
    trainable = [str(m).strip() for m in (model_cfg.get("trainable_backbone_modules", []) or []) if str(m).strip()]
    freeze_backbone = bool(model_cfg.get("freeze_backbone", True))
    train_openfold_all = bool(model_cfg.get("train_openfold_all", False))

    resolved_ckpt = _resolve_openfold_checkpoint(model_cfg, base_dir=base_dir)
    model_cfg["openfold_checkpoint"] = resolved_ckpt

    if freeze_backbone and not train_openfold_all and not trainable and not resolved_ckpt:
        raise ValueError(
            "Invalid config: model.freeze_backbone=true with no trainable backbone modules and no "
            "model.openfold_checkpoint. This freezes a randomly initialized backbone. "
            "Set model.openfold_checkpoint, enable model.train_openfold_all, or set "
            "model.trainable_backbone_modules."
        )

    out["model"] = model_cfg
    return out


def _build_trainer_logger(cfg: Dict, output_dir: Path):
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return True

    api_key = wandb_cfg.get("api_key", None)
    api_key_env = wandb_cfg.get("api_key_env", "WANDB_API_KEY")

    if api_key:
        os.environ["WANDB_API_KEY"] = str(api_key)
    elif api_key_env and api_key_env in os.environ:
        logger.info("Using W&B API key from env var %s", api_key_env)
    else:
        logger.warning(
            "W&B enabled but no API key provided via wandb.api_key or env var %s. "
            "Proceeding with existing wandb login state.",
            api_key_env,
        )

    try:
        from pytorch_lightning.loggers import WandbLogger
    except Exception as exc:
        raise SystemExit(
            "W&B logging requested but WandbLogger is unavailable. Install wandb in the active environment."
        ) from exc

    kwargs = {
        "project": wandb_cfg.get("project", "openfold-retrieval"),
        "entity": wandb_cfg.get("entity", None),
        "name": wandb_cfg.get("run_name", None),
        "save_dir": str(output_dir),
        "offline": bool(wandb_cfg.get("offline", False)),
    }

    run_id = wandb_cfg.get("run_id", None)
    if run_id:
        kwargs["id"] = run_id
        kwargs["resume"] = wandb_cfg.get("resume", "allow")

    tags = wandb_cfg.get("tags", None)
    if tags:
        if isinstance(tags, str):
            kwargs["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
        elif isinstance(tags, list):
            kwargs["tags"] = [str(t) for t in tags]

    wb_logger = WandbLogger(**kwargs)
    logger.info(
        "W&B logger enabled (project=%s, entity=%s, run_id=%s, offline=%s)",
        kwargs.get("project"),
        kwargs.get("entity"),
        kwargs.get("id"),
        kwargs.get("offline"),
    )
    return wb_logger


def _build_data_module(cfg: Dict, model_cfg: Dict, retrieval_cfg: Dict) -> "RetrievalDataModule":
    from scripts.retrieval.retrieval_data import RetrievalDataModule, build_manifest, write_manifest_jsonl

    data_cfg = cfg.get("data", {})
    retrieval_pipeline = str(retrieval_cfg.get("pipeline", "legacy")).strip().lower()

    packed_dataset_dir = data_cfg.get("packed_dataset_dir", None)
    dataset_dir = data_cfg.get("dataset_dir", None)
    manifest_path = data_cfg.get("manifest_path", None)
    write_manifest_to = data_cfg.get("write_manifest_to", None)

    if retrieval_pipeline in {"embed_project", "rawseq_esm1b_ragstyle"} and packed_dataset_dir is not None:
        raise ValueError(
            f"retrieval.pipeline={retrieval_pipeline} requires `raw_sequence` metadata in each batch. "
            "Packed retrieval shards currently store tensor features only. "
            "Use data.dataset_dir or data.manifest_path."
        )

    if dataset_dir and write_manifest_to:
        records = build_manifest(Path(dataset_dir), strict=True)
        write_manifest_jsonl(records, Path(write_manifest_to))
        logger.info("Wrote manifest to %s", write_manifest_to)

    dm = RetrievalDataModule(
        packed_dataset_dir=Path(packed_dataset_dir) if packed_dataset_dir else None,
        dataset_dir=Path(dataset_dir) if dataset_dir else None,
        manifest_path=Path(manifest_path) if manifest_path else None,
        seq_embedding_dir=Path(data_cfg["seq_embedding_dir"]) if data_cfg.get("seq_embedding_dir") else None,
        config_preset=str(data_cfg.get("config_preset", model_cfg.get("config_preset", "seqemb_initial_training"))),
        batch_size=int(data_cfg.get("batch_size", 1)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        low_prec=bool(model_cfg.get("low_prec", False)),
        disable_templates=bool(data_cfg.get("disable_templates", True)),
        max_recycling_iters=int(data_cfg.get("max_recycling_iters", 0)),
        strict_seq_embeddings=bool(data_cfg.get("strict_seq_embeddings", False)),
        seq_embedding_dim=int(retrieval_cfg.get("seq_embedding_dim", 1280)),
    )
    return dm


def _build_model(cfg: Dict) -> "RetrievalAugmentedLightningModule":
    from openfold.model.retrieval.lightning_module import RetrievalAugmentedLightningModule

    model_cfg = cfg.get("model", {})
    retrieval_cfg = cfg.get("retrieval", {})
    fusion_cfg = retrieval_cfg.get("fusion", {})
    source_cfg = retrieval_cfg.get("sources", {})
    seq_source = source_cfg.get("seq", {})
    struct_source = source_cfg.get("struct", {})
    optimizer_cfg = cfg.get("optimizer", {})

    embed_project_cfg = retrieval_cfg.get("embed_project", {})
    rawseq_cfg = retrieval_cfg.get("rawseq_esm1b", {})
    lora_targets = rawseq_cfg.get(
        "lora_target_modules",
        ["self_attn.q_proj", "self_attn.v_proj", "self_attn.out_proj", "fc1", "fc2"],
    )
    if isinstance(lora_targets, str):
        lora_targets = [t.strip() for t in lora_targets.split(",") if t.strip()]

    return RetrievalAugmentedLightningModule(
        config_preset=str(model_cfg.get("config_preset", "seqemb_initial_training")),
        seq_embedding_dim=int(retrieval_cfg.get("seq_embedding_dim", 1280)),
        top_k=int(retrieval_cfg.get("top_k", 8)),
        lr=float(optimizer_cfg.get("lr", 1e-4)),
        struct_index_path=struct_source.get("index_path", None),
        struct_index_dim=int(struct_source.get("index_dim", 512)),
        seq_index_path=seq_source.get("index_path", None),
        seq_index_dim=int(seq_source.get("index_dim", 1280)),
        nprobe=int(retrieval_cfg.get("nprobe", 64)),
        retrieval_ablation=str(retrieval_cfg.get("ablation", "both")),
        retrieval_pipeline=str(retrieval_cfg.get("pipeline", "legacy")),
        fusion_name=str(fusion_cfg.get("name", "simple_cross_attn")),
        fusion_params=dict(fusion_cfg.get("params", {})),
        retrieval_injection_stages=tuple(retrieval_cfg.get("injection", {}).get("stages", ["input"])),
        retriever_esm2_model_name=str(embed_project_cfg.get("retriever_esm2_model_name", "esm2_t12_35M_UR50D")),
        retriever_esm2_repr_layer=int(embed_project_cfg.get("retriever_esm2_repr_layer", 12)),
        retriever_esm2_max_len=int(embed_project_cfg.get("retriever_esm2_max_len", 1022)),
        retriever_tmvec_checkpoint=embed_project_cfg.get("retriever_tmvec_checkpoint", None),
        retriever_tmvec_max_len=int(embed_project_cfg.get("retriever_tmvec_max_len", 1022)),
        retriever_normalize_queries=bool(embed_project_cfg.get("retriever_normalize_queries", True)),
        rawseq_seq_index_ids_path=rawseq_cfg.get("seq_index_ids_path", None),
        rawseq_seq_db_fasta_path=rawseq_cfg.get("seq_db_fasta_path", None),
        rawseq_seq_db_fasta_index_db=rawseq_cfg.get("seq_db_fasta_index_db", None),
        rawseq_esm1b_model_name=str(rawseq_cfg.get("esm1b_model_name", "esm1b_t33_650M_UR50S")),
        rawseq_esm1b_repr_layer=int(rawseq_cfg.get("esm1b_repr_layer", 33)),
        rawseq_esm1b_max_len=int(rawseq_cfg.get("esm1b_max_len", 1022)),
        rawseq_esm1b_tuning_mode=str(rawseq_cfg.get("esm1b_tuning_mode", "lora")),
        rawseq_esm1b_lora_rank=int(rawseq_cfg.get("lora_rank", 8)),
        rawseq_esm1b_lora_alpha=float(rawseq_cfg.get("lora_alpha", 16.0)),
        rawseq_esm1b_lora_dropout=float(rawseq_cfg.get("lora_dropout", 0.0)),
        rawseq_esm1b_lora_target_modules=tuple(lora_targets),
        rawseq_esm1b_train_layer_norm=bool(rawseq_cfg.get("train_layer_norm", False)),
        rawseq_esm1b_backend=str(rawseq_cfg.get("backend", "fair_esm")),
        rawseq_esm1b_use_pretrained=bool(rawseq_cfg.get("use_pretrained", True)),
        rawseq_esm1b_compute_dtype=str(rawseq_cfg.get("compute_dtype", "bfloat16")),
        openfold_checkpoint=model_cfg.get("openfold_checkpoint", None),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
        train_openfold_all=bool(model_cfg.get("train_openfold_all", False)),
        trainable_backbone_modules=tuple(model_cfg.get("trainable_backbone_modules", [])),
        controller_lr=optimizer_cfg.get("controller_lr", None),
        backbone_lr=optimizer_cfg.get("backbone_lr", None),
        auto_disable_resolution_gated_losses=bool(model_cfg.get("auto_disable_resolution_gated_losses", True)),
        resolution_gated_loss_warmup_steps=int(model_cfg.get("resolution_gated_loss_warmup_steps", 128)),
        low_prec=bool(model_cfg.get("low_prec", False)),
        openfold_use_flash=bool(model_cfg.get("use_flash", False)),
    )


def main():
    parser = argparse.ArgumentParser(description="Train modular retrieval-augmented OpenFold from YAML config")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values as dotted.key=value (repeatable)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = _load_config(args.config)
    cfg = _apply_overrides(cfg, args.overrides)
    cfg = _validate_model_cfg(cfg, base_dir=REPO_ROOT)

    train_cfg = cfg.get("train", {})
    seed = train_cfg.get("seed", None)
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)

    model = _build_model(cfg)

    model_cfg = cfg.get("model", {})
    retrieval_cfg = cfg.get("retrieval", {})
    data_module = _build_data_module(cfg, model_cfg=model_cfg, retrieval_cfg=retrieval_cfg)

    trainer_cfg = cfg.get("trainer", {})
    output_dir = Path(trainer_cfg.get("output_dir", "./outputs/retrieval_modular"))
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer_logger = _build_trainer_logger(cfg, output_dir=output_dir)
    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        logger=trainer_logger,
        accelerator=str(trainer_cfg.get("accelerator", _auto_accelerator())),
        devices=int(trainer_cfg.get("devices", 1)),
        precision=str(trainer_cfg.get("precision", "32")),
        max_epochs=int(trainer_cfg.get("max_epochs", 1)),
        val_check_interval=int(trainer_cfg.get("val_check_interval", 1000)),
        accumulate_grad_batches=int(trainer_cfg.get("accumulate_grad_batches", 1)),
        num_sanity_val_steps=int(trainer_cfg.get("num_sanity_val_steps", 0)),
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 25)),
        enable_checkpointing=bool(trainer_cfg.get("enable_checkpointing", True)),
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
