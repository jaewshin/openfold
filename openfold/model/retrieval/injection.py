from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn

from .interfaces import InjectionStrategy


class RetrievalInjectionPlan(InjectionStrategy):
    """Configurable multi-stage retrieval injection into OpenFold features."""

    _VALID_STAGES = ("input", "pre_evoformer", "pre_structure")

    def __init__(
        self,
        seq_embedding_dim: int,
        c_m: int,
        c_s: int,
        stages: Iterable[str] = ("input",),
    ):
        super().__init__()
        stage_values = {str(s).strip().lower() for s in stages if str(s).strip()}
        if not stage_values:
            stage_values = {"input"}
        invalid = sorted(stage_values - set(self._VALID_STAGES))
        if invalid:
            raise ValueError(
                f"Unsupported retrieval injection stage(s): {invalid}. "
                f"Expected subset of {list(self._VALID_STAGES)}"
            )

        self.stages = tuple(sorted(stage_values))
        self.inject_input_embedding = "input" in stage_values
        self.inject_pre_evoformer = "pre_evoformer" in stage_values
        self.inject_pre_structure = "pre_structure" in stage_values

        self.input_gate = nn.Parameter(torch.tensor(0.0)) if self.inject_input_embedding else None
        self.pre_evoformer_gate = nn.Parameter(torch.tensor(0.0)) if self.inject_pre_evoformer else None
        self.pre_structure_gate = nn.Parameter(torch.tensor(0.0)) if self.inject_pre_structure else None

        self.pre_evoformer_proj = (
            nn.Linear(seq_embedding_dim, c_m) if self.inject_pre_evoformer else None
        )
        self.pre_structure_proj = (
            nn.Linear(seq_embedding_dim, c_s) if self.inject_pre_structure else None
        )

    @property
    def ordered_stages(self) -> Tuple[str, ...]:
        """Canonical application order for deterministic behavior."""
        order = []
        for s in self._VALID_STAGES:
            if s in self.stages:
                order.append(s)
        return tuple(order)

    def apply(
        self,
        tensor_batch: Dict[str, torch.Tensor],
        fused_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        model_batch = dict(tensor_batch)
        num_recycles = tensor_batch["seq_embedding"].shape[-1]
        current_tokens = tensor_batch["seq_embedding"][..., 0]

        for stage in self.ordered_stages:
            if stage == "input":
                assert self.input_gate is not None
                gate = torch.sigmoid(self.input_gate)
                blended = current_tokens + gate * (fused_tokens - current_tokens)
                model_batch["seq_embedding"] = blended.unsqueeze(-1).expand(*blended.shape, num_recycles)
                current_tokens = blended
            elif stage == "pre_evoformer":
                assert self.pre_evoformer_gate is not None
                assert self.pre_evoformer_proj is not None
                delta_m = self.pre_evoformer_proj(fused_tokens)
                delta_m = torch.sigmoid(self.pre_evoformer_gate) * delta_m
                model_batch["retrieval_pre_evoformer"] = delta_m.unsqueeze(-1).expand(
                    *delta_m.shape,
                    num_recycles,
                )
            elif stage == "pre_structure":
                assert self.pre_structure_gate is not None
                assert self.pre_structure_proj is not None
                delta_s = self.pre_structure_proj(fused_tokens)
                delta_s = torch.sigmoid(self.pre_structure_gate) * delta_s
                model_batch["retrieval_pre_structure"] = delta_s.unsqueeze(-1).expand(
                    *delta_s.shape,
                    num_recycles,
                )
            else:
                raise ValueError(f"Unsupported stage: {stage}")

        return model_batch
