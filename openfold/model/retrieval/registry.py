from __future__ import annotations

from typing import Callable, Dict, Iterable

from .interfaces import FusionStrategy


_FUSION_REGISTRY: Dict[str, Callable[..., FusionStrategy]] = {}


def register_fusion(name: str):
    """Decorator for registering fusion strategy constructors."""

    key = str(name).strip().lower()
    if not key:
        raise ValueError("Fusion name must be non-empty")

    def _decorator(factory: Callable[..., FusionStrategy]):
        if key in _FUSION_REGISTRY:
            raise ValueError(f"Fusion strategy already registered: {key}")
        _FUSION_REGISTRY[key] = factory
        return factory

    return _decorator


def build_fusion(name: str, **kwargs) -> FusionStrategy:
    key = str(name).strip().lower()
    if key not in _FUSION_REGISTRY:
        available = ", ".join(sorted(_FUSION_REGISTRY.keys()))
        raise KeyError(f"Unknown fusion strategy '{name}'. Available: [{available}]")
    return _FUSION_REGISTRY[key](**kwargs)


def available_fusions() -> Iterable[str]:
    return tuple(sorted(_FUSION_REGISTRY.keys()))
