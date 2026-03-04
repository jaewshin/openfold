from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Sequence, Set

import pytorch_lightning as pl
import torch
import torch.nn as nn

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.loss import AlphaFoldLoss
from openfold.utils.tensor_utils import tensor_tree_map

# Register built-in fusion strategies.
from . import fusion_rag_esm_inspired as _fusion_rag_esm_inspired  # noqa: F401
from . import fusion_simple_cross_attn as _fusion_simple_cross_attn  # noqa: F401
from .controller import RetrievalController
from .injection import RetrievalInjectionPlan
from .pipeline_embed_project import EmbedProjectQueryPipeline
from .pipeline_legacy import LegacyQueryPipeline
from .registry import build_fusion
from .retriever import LazyFaissRetriever

logger = logging.getLogger(__name__)


class RetrievalAugmentedLightningModule(pl.LightningModule):
    """Modular retrieval-augmented OpenFold training module."""

    def __init__(
        self,
        config_preset: str = "seqemb_initial_training",
        seq_embedding_dim: int = 1280,
        top_k: int = 8,
        lr: float = 1e-4,
        struct_index_path: Optional[str] = None,
        struct_index_dim: int = 512,
        seq_index_path: Optional[str] = None,
        seq_index_dim: int = 1280,
        nprobe: int = 64,
        retrieval_ablation: str = "both",
        retrieval_pipeline: str = "legacy",
        fusion_name: str = "simple_cross_attn",
        fusion_params: Optional[Dict[str, object]] = None,
        retrieval_injection_stages: Sequence[str] = ("input",),
        retriever_esm2_model_name: str = "esm2_t12_35M_UR50D",
        retriever_esm2_repr_layer: int = 12,
        retriever_esm2_max_len: int = 1022,
        retriever_tmvec_checkpoint: Optional[str] = None,
        retriever_tmvec_max_len: int = 1022,
        retriever_normalize_queries: bool = True,
        openfold_checkpoint: Optional[str] = None,
        freeze_backbone: bool = True,
        train_openfold_all: bool = False,
        trainable_backbone_modules: Sequence[str] = (),
        controller_lr: Optional[float] = None,
        backbone_lr: Optional[float] = None,
        auto_disable_resolution_gated_losses: bool = True,
        resolution_gated_loss_warmup_steps: int = 128,
        low_prec: bool = False,
        backbone_factory: Optional[Callable] = None,
        loss_factory: Optional[Callable] = None,
        seq_query_encoder_override: Optional[nn.Module] = None,
        struct_query_encoder_override: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "backbone_factory",
                "loss_factory",
                "seq_query_encoder_override",
                "struct_query_encoder_override",
            ]
        )

        if retrieval_pipeline not in {"legacy", "embed_project"}:
            raise ValueError(
                f"Unsupported retrieval_pipeline={retrieval_pipeline!r}. "
                "Expected one of {'legacy', 'embed_project'}."
            )
        if retrieval_ablation not in {"both", "seq_only", "struct_only"}:
            raise ValueError(
                f"Unsupported retrieval_ablation={retrieval_ablation!r}. "
                "Expected one of {'both', 'seq_only', 'struct_only'}."
            )

        self.lr = float(lr)
        self.controller_lr = float(controller_lr) if controller_lr is not None else None
        self.backbone_lr = float(backbone_lr) if backbone_lr is not None else None
        self.freeze_backbone = bool(freeze_backbone)
        self.train_openfold_all = bool(train_openfold_all)
        self.retrieval_pipeline = retrieval_pipeline
        self.auto_disable_resolution_gated_losses = bool(auto_disable_resolution_gated_losses)
        self.resolution_gated_loss_warmup_steps = max(1, int(resolution_gated_loss_warmup_steps))
        self._resolution_samples_seen = 0
        self._resolution_valid_counts: Dict[str, int] = {}
        self._resolution_autodisabled: Set[str] = set()
        self._resolution_term_bounds: Dict[str, Sequence[float]] = {}
        self._nan_skip_total = 0

        self.config = model_config(config_preset, train=True, low_prec=low_prec)
        # Retrieval training fixtures do not require templates.
        self.config.model.template.enabled = False
        self.config.data.common.use_templates = False

        self.c_m = int(self.config.model.evoformer_stack.c_m)
        self.c_s = int(self.config.model.evoformer_stack.c_s)

        requested_trainable_modules = tuple(str(m).strip() for m in trainable_backbone_modules if str(m).strip())
        if self.freeze_backbone and not self.train_openfold_all and not requested_trainable_modules and not openfold_checkpoint:
            raise ValueError(
                "freeze_backbone=True with no trainable OpenFold modules requires openfold_checkpoint. "
                "Otherwise the frozen backbone remains randomly initialized."
            )

        self.openfold = AlphaFold(self.config) if backbone_factory is None else backbone_factory(self.config)

        if openfold_checkpoint:
            self._load_openfold_checkpoint(openfold_checkpoint)

        self._openfold_trainable_modules = self._configure_backbone_trainability(
            freeze_backbone=self.freeze_backbone,
            train_openfold_all=self.train_openfold_all,
            trainable_backbone_modules=requested_trainable_modules,
        )

        if retrieval_pipeline == "legacy":
            query_pipeline = LegacyQueryPipeline(
                seq_embedding_dim=seq_embedding_dim,
                seq_index_dim=seq_index_dim,
                struct_index_dim=struct_index_dim,
            )
        else:
            query_pipeline = EmbedProjectQueryPipeline(
                use_seq_encoder=seq_index_path is not None,
                use_struct_encoder=struct_index_path is not None,
                retriever_esm2_model_name=retriever_esm2_model_name,
                retriever_esm2_repr_layer=retriever_esm2_repr_layer,
                retriever_esm2_max_len=retriever_esm2_max_len,
                retriever_tmvec_checkpoint=retriever_tmvec_checkpoint,
                retriever_tmvec_max_len=retriever_tmvec_max_len,
                retriever_normalize_queries=retriever_normalize_queries,
                seq_encoder_override=seq_query_encoder_override,
                struct_encoder_override=struct_query_encoder_override,
            )

        fusion_cfg = dict(fusion_params or {})
        fusion_cfg.setdefault("emb_dim", seq_embedding_dim)
        seq_fusion = build_fusion(fusion_name, **fusion_cfg)
        struct_fusion = build_fusion(fusion_name, **fusion_cfg)

        injection_plan = RetrievalInjectionPlan(
            seq_embedding_dim=seq_embedding_dim,
            c_m=self.c_m,
            c_s=self.c_s,
            stages=retrieval_injection_stages,
        )

        seq_retriever = LazyFaissRetriever(seq_index_path, top_k=top_k, nprobe=nprobe) if seq_index_path else None
        struct_retriever = (
            LazyFaissRetriever(struct_index_path, top_k=top_k, nprobe=nprobe) if struct_index_path else None
        )

        seq_db_proj = nn.Linear(seq_index_dim, seq_embedding_dim)
        struct_db_proj = nn.Linear(struct_index_dim, seq_embedding_dim)

        self.controller = RetrievalController(
            openfold=self.openfold,
            query_pipeline=query_pipeline,
            seq_fusion=seq_fusion,
            struct_fusion=struct_fusion,
            injection_plan=injection_plan,
            top_k=top_k,
            retrieval_ablation=retrieval_ablation,
            seq_retriever=seq_retriever,
            struct_retriever=struct_retriever,
            seq_db_proj=seq_db_proj,
            struct_db_proj=struct_db_proj,
        )

        self.loss_fn = AlphaFoldLoss(self.config.loss) if loss_factory is None else loss_factory(self.config.loss)
        self._init_resolution_gated_tracking()

    def _load_openfold_checkpoint(self, checkpoint_path: str) -> None:
        sd = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]

        remapped = {}
        for k, v in sd.items():
            if k.startswith("model.openfold."):
                remapped[k[len("model.openfold.") :]] = v
            elif k.startswith("model."):
                remapped[k[len("model.") :]] = v
            else:
                remapped[k] = v

        missing, unexpected = self.openfold.load_state_dict(remapped, strict=False)
        logger.info(
            "Loaded OpenFold checkpoint=%s (missing=%d unexpected=%d)",
            checkpoint_path,
            len(missing),
            len(unexpected),
        )

    def _configure_backbone_trainability(
        self,
        freeze_backbone: bool,
        train_openfold_all: bool,
        trainable_backbone_modules: Sequence[str],
    ) -> Sequence[str]:
        requested: Set[str] = {str(m).strip() for m in trainable_backbone_modules if str(m).strip()}

        if freeze_backbone:
            for p in self.openfold.parameters():
                p.requires_grad_(False)
            self.openfold.eval()

            if train_openfold_all:
                for p in self.openfold.parameters():
                    p.requires_grad_(True)
                self.openfold.train()
                logger.info("OpenFold backbone fully trainable (train_openfold_all=True).")
                return []

            enabled = []
            missing = []
            for name in sorted(requested):
                module = getattr(self.openfold, name, None)
                if module is None:
                    missing.append(name)
                    continue
                for p in module.parameters():
                    p.requires_grad_(True)
                module.train()
                enabled.append(name)

            if missing:
                logger.warning(
                    "Requested OpenFold trainable modules not found and ignored: %s",
                    ", ".join(missing),
                )
            if enabled:
                logger.info("OpenFold backbone frozen except modules: %s", ", ".join(enabled))
            return enabled

        if requested:
            logger.info(
                "freeze_backbone=False: full OpenFold backbone is trainable; "
                "module-specific train flags are redundant."
            )
        return sorted(requested)

    def on_train_start(self):
        # Keep frozen blocks in eval mode while selected modules remain trainable.
        if self.freeze_backbone:
            if self.train_openfold_all:
                self.openfold.train()
                return
            self.openfold.eval()
            for module_name in self._openfold_trainable_modules:
                module = getattr(self.openfold, module_name, None)
                if module is not None:
                    module.train()

    def forward(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        return self.controller(batch)

    @staticmethod
    def _term_is_resolution_gated(term_cfg) -> bool:
        return hasattr(term_cfg, "min_resolution") and hasattr(term_cfg, "max_resolution")

    def _init_resolution_gated_tracking(self) -> None:
        self._resolution_valid_counts = {}
        self._resolution_term_bounds = {}
        if not hasattr(self.loss_fn, "config"):
            return
        for term_name in self.loss_fn.config.keys():
            term_cfg = self.loss_fn.config[term_name]
            if not self._term_is_resolution_gated(term_cfg):
                continue
            self._resolution_valid_counts[term_name] = 0
            self._resolution_term_bounds[term_name] = (float(term_cfg.min_resolution), float(term_cfg.max_resolution))

    @staticmethod
    def _last_recycle_view(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim <= 1:
            return tensor
        return tensor[..., -1]

    def _track_resolution_gated_coverage(self, batch: Dict[str, object]) -> None:
        resolution = batch.get("resolution", None)
        if not torch.is_tensor(resolution):
            return

        resolution_last = self._last_recycle_view(resolution.detach().float())
        resolution_flat = resolution_last.reshape(-1)
        if resolution_flat.numel() == 0:
            return

        self._resolution_samples_seen += int(resolution_flat.numel())
        self.log(
            "train/resolution_mean",
            resolution_flat.mean(),
            on_step=True,
            on_epoch=False,
            logger=True,
        )

        for term_name, (min_res, max_res) in self._resolution_term_bounds.items():
            term_cfg = self.loss_fn.config[term_name]
            weight = float(term_cfg.weight)
            valid_mask = (resolution_flat >= min_res) & (resolution_flat <= max_res)
            valid_count = int(valid_mask.sum().item())
            self._resolution_valid_counts[term_name] += valid_count
            self.log(
                f"train/{term_name}_resolution_valid_frac",
                valid_mask.float().mean(),
                on_step=True,
                on_epoch=False,
                logger=True,
            )

            if (
                self.auto_disable_resolution_gated_losses
                and weight > 0.0
                and term_name not in self._resolution_autodisabled
                and self._resolution_samples_seen >= self.resolution_gated_loss_warmup_steps
                and self._resolution_valid_counts[term_name] == 0
            ):
                term_cfg.weight = 0.0
                self._resolution_autodisabled.add(term_name)
                logger.warning(
                    "Auto-disabled loss term '%s' after %d resolution samples with no values in [%s, %s].",
                    term_name,
                    self._resolution_samples_seen,
                    min_res,
                    max_res,
                )
                self.log(
                    f"train/{term_name}_auto_disabled",
                    torch.tensor(1.0, device=resolution_flat.device),
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                )

    def _log_nonfinite_output_flags(self, outputs: Dict[str, object]) -> None:
        flag = 0.0
        for value in outputs.values():
            if torch.is_tensor(value):
                if not torch.isfinite(value).all():
                    flag = 1.0
                    break
                continue
            if isinstance(value, dict):
                for nested in value.values():
                    if torch.is_tensor(nested) and not torch.isfinite(nested).all():
                        flag = 1.0
                        break
                if flag > 0.0:
                    break
            if isinstance(value, (list, tuple)):
                for nested in value:
                    if torch.is_tensor(nested) and not torch.isfinite(nested).all():
                        flag = 1.0
                        break
                if flag > 0.0:
                    break

        self.log(
            "train/nonfinite_output_flag",
            torch.tensor(flag, device=self.device),
            on_step=True,
            on_epoch=False,
            logger=True,
        )

    def training_step(self, batch: Dict[str, object], batch_idx: int):
        del batch_idx
        self._track_resolution_gated_coverage(batch)
        outputs = self(batch)
        self._log_nonfinite_output_flags(outputs)
        tensor_batch = {k: v for k, v in batch.items() if torch.is_tensor(v)}
        labels = tensor_tree_map(lambda t: t[..., -1], tensor_batch)
        loss, breakdown = self.loss_fn(outputs, labels, _return_breakdown=True)

        step_nan_skips = 0
        for name, value in breakdown.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=False, logger=True)
            if name.endswith("_nan_skipped"):
                step_nan_skips += int(value.item())
        self._nan_skip_total += step_nan_skips
        self.log(
            "train/nan_skipped_terms",
            torch.tensor(float(step_nan_skips), device=loss.device),
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        self.log(
            "train/nan_skipped_terms_cum",
            torch.tensor(float(self._nan_skip_total), device=loss.device),
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        self.log(
            "train/seq_retrieval_entropy",
            -(
                outputs["seq_retrieval_scores"]
                * outputs["seq_retrieval_scores"].clamp(min=1e-8).log()
            ).sum(dim=-1).mean(),
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        self.log(
            "train/struct_retrieval_entropy",
            -(
                outputs["struct_retrieval_scores"]
                * outputs["struct_retrieval_scores"].clamp(min=1e-8).log()
            ).sum(dim=-1).mean(),
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        self.log(
            "train/source_seq_weight",
            outputs["retrieval_source_weights"][0],
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        self.log(
            "train/source_struct_weight",
            outputs["retrieval_source_weights"][1],
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, object], batch_idx: int):
        del batch_idx
        outputs = self(batch)
        tensor_batch = {k: v for k, v in batch.items() if torch.is_tensor(v)}
        labels = tensor_tree_map(lambda t: t[..., -1], tensor_batch)
        labels["use_clamped_fape"] = 0.0
        _, breakdown = self.loss_fn(outputs, labels, _return_breakdown=True)
        for name, value in breakdown.items():
            self.log(f"val/{name}", value, on_step=False, on_epoch=True, logger=True, sync_dist=False)

    def configure_optimizers(self):
        controller_params = []
        backbone_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("openfold."):
                backbone_params.append(param)
            else:
                controller_params.append(param)

        param_groups = []
        controller_lr = self.lr if self.controller_lr is None else self.controller_lr
        backbone_lr = self.lr if self.backbone_lr is None else self.backbone_lr

        if controller_params:
            param_groups.append({"params": controller_params, "lr": float(controller_lr), "name": "controller"})
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": float(backbone_lr), "name": "openfold_backbone"})

        if not param_groups:
            raise RuntimeError("No trainable parameters found when configuring optimizer.")

        logger.info(
            "Optimizer param groups: controller=%d (lr=%.3e), backbone=%d (lr=%.3e)",
            len(controller_params),
            float(controller_lr),
            len(backbone_params),
            float(backbone_lr),
        )
        return torch.optim.Adam(param_groups, eps=1e-5)
