"""Modular retrieval components for retrieval-augmented OpenFold training."""

# Import fusion modules for registry side-effects.
from . import fusion_rag_esm_inspired as _fusion_rag_esm_inspired  # noqa: F401
from . import fusion_simple_cross_attn as _fusion_simple_cross_attn  # noqa: F401
from .controller import RetrievalController
from .injection import RetrievalInjectionPlan
from .interfaces import FusionStrategy, InjectionStrategy, QueryPipeline, QueryVectors
from .pipeline_embed_project import EmbedProjectQueryPipeline
from .pipeline_legacy import LegacyQueryPipeline
from .registry import available_fusions, build_fusion, register_fusion
from .retriever import LazyFaissRetriever

try:
    from .lightning_module import RetrievalAugmentedLightningModule
except Exception:
    RetrievalAugmentedLightningModule = None  # type: ignore

__all__ = [
    "FusionStrategy",
    "InjectionStrategy",
    "QueryPipeline",
    "QueryVectors",
    "RetrievalController",
    "RetrievalInjectionPlan",
    "EmbedProjectQueryPipeline",
    "LegacyQueryPipeline",
    "LazyFaissRetriever",
    "register_fusion",
    "build_fusion",
    "available_fusions",
    "RetrievalAugmentedLightningModule",
]
