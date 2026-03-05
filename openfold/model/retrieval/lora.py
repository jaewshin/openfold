from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """LoRA adapter for nn.Linear with frozen base weights."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected nn.Linear base module, got {type(base).__name__}")
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")

        self.base = base
        # Preserve nn.Linear-like interface for callsites that directly read
        # `weight`/`bias` or `in_features`/`out_features`.
        self.in_features = int(self.base.in_features)
        self.out_features = int(self.base.out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(self.rank, self.base.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = torch.matmul(self.dropout(x), self.lora_A.t())
        delta = torch.matmul(delta, self.lora_B.t())
        return base_out + self.scaling * delta


def _resolve_parent_module(root: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    if not parts:
        raise ValueError(f"Invalid module name: {qualified_name!r}")

    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]  # type: ignore[index]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def _assign_child_module(parent: nn.Module, child_name: str, module: nn.Module) -> None:
    if child_name.isdigit():
        parent[int(child_name)] = module  # type: ignore[index]
    else:
        setattr(parent, child_name, module)


def apply_lora_to_linear_modules(
    model: nn.Module,
    target_substrings: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> List[str]:
    """Wrap target linear modules with LoRA adapters.

    Returns:
        List of fully-qualified module names patched with LoRA.
    """
    targets = tuple(s for s in (str(t).strip() for t in target_substrings) if s)
    if not targets:
        return []

    to_patch: List[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, LoRALinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if any(token in name for token in targets):
            to_patch.append(name)

    patched: List[str] = []
    for name in to_patch:
        parent, child = _resolve_parent_module(model, name)
        current = parent[int(child)] if child.isdigit() else getattr(parent, child)
        if not isinstance(current, nn.Linear):
            continue
        _assign_child_module(
            parent,
            child,
            LoRALinear(
                base=current,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )
        patched.append(name)

    return patched


def count_trainable_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
