"""Modular retrieval components for retrieval-augmented OpenFold training."""

# Import fusion modules for registry side-effects.
from . import fusion_rag_esm_inspired as _fusion_rag_esm_inspired  # noqa: F401
from . import fusion_rag_esm_port as _fusion_rag_esm_port  # noqa: F401
from . import fusion_simple_cross_attn as _fusion_simple_cross_attn  # noqa: F401
from .context_esm1b import ESM1bContextEncoder
from .controller import RetrievalController
from .injection import RetrievalInjectionPlan
from .interfaces import FusionStrategy, InjectionStrategy, QueryPipeline, QueryVectors
from .pipeline_embed_project import EmbedProjectQueryPipeline
from .pipeline_legacy import LegacyQueryPipeline
from .row_id_lookup import RowIdLookup
from .registry import available_fusions, build_fusion, register_fusion
from .sequence_store import FastaSequenceStore
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
    "RowIdLookup",
    "FastaSequenceStore",
    "ESM1bContextEncoder",
    "LazyFaissRetriever",
    "register_fusion",
    "build_fusion",
    "available_fusions",
    "RetrievalAugmentedLightningModule",
]
