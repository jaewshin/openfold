from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .interfaces import FusionStrategy, InjectionStrategy, QueryPipeline
from .retriever import LazyFaissRetriever, search_index


class RetrievalController(nn.Module):
    """Composable retrieval controller for retrieval-augmented OpenFold."""

    def __init__(
        self,
        openfold: nn.Module,
        query_pipeline: QueryPipeline,
        seq_fusion: FusionStrategy,
        struct_fusion: FusionStrategy,
        injection_plan: InjectionStrategy,
        top_k: int = 8,
        retrieval_ablation: str = "both",
        seq_retriever: Optional[LazyFaissRetriever] = None,
        struct_retriever: Optional[LazyFaissRetriever] = None,
        seq_db_proj: Optional[nn.Linear] = None,
        struct_db_proj: Optional[nn.Linear] = None,
    ):
        super().__init__()
        if retrieval_ablation not in {"both", "seq_only", "struct_only"}:
            raise ValueError(
                f"Unsupported retrieval_ablation={retrieval_ablation!r}. "
                "Expected one of {'both', 'seq_only', 'struct_only'}."
            )

        self.openfold = openfold
        self.query_pipeline = query_pipeline
        self.seq_fusion = seq_fusion
        self.struct_fusion = struct_fusion
        self.injection_plan = injection_plan

        self.top_k = int(top_k)
        self.retrieval_ablation = retrieval_ablation

        self.seq_retriever = seq_retriever
        self.struct_retriever = struct_retriever
        self.seq_db_proj = seq_db_proj
        self.struct_db_proj = struct_db_proj

        self.source_mix_logits = nn.Parameter(torch.zeros(2))

    def _enabled_sources(self) -> Tuple[bool, bool]:
        use_seq = self.retrieval_ablation in {"both", "seq_only"}
        use_struct = self.retrieval_ablation in {"both", "struct_only"}

        if use_seq and self.seq_retriever is None:
            use_seq = False
        if use_struct and self.struct_retriever is None:
            use_struct = False

        if not use_seq and not use_struct:
            raise ValueError(
                "No active retriever source. Provide seq/struct retrievers or adjust retrieval_ablation."
            )

        return use_seq, use_struct

    def _run_source(
        self,
        source: str,
        query_tokens: torch.Tensor,
        query_vec: torch.Tensor,
        retriever: LazyFaissRetriever,
        db_proj: nn.Linear,
        fusion: FusionStrategy,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores, indices, retrieved_keys, valid = search_index(
            query_vec=query_vec,
            retriever=retriever,
            top_k=self.top_k,
        )

        # Reconstructed vectors are per-hit vectors; represent each as 1 token.
        retrieved_tokens = db_proj(retrieved_keys.unsqueeze(1))
        retrieved_masks = valid.float().unsqueeze(-1)
        fused = fusion(
            query_tokens,
            retrieved_tokens,
            scores,
            retrieved_masks=retrieved_masks,
            context={"source": source},
        )
        return fused, scores, indices

    def forward(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        tensor_batch = {k: v for k, v in batch.items() if torch.is_tensor(v)}
        metadata_batch = {k: v for k, v in batch.items() if not torch.is_tensor(v)}

        query_tokens = tensor_batch["seq_embedding"][..., 0]  # [B, N, D]
        query_mask = tensor_batch.get("seq_mask", None)
        if query_mask is not None:
            query_mask = query_mask[..., 0]

        raw_sequences = metadata_batch.get("raw_sequence", None)
        use_seq, use_struct = self._enabled_sources()
        vectors = self.query_pipeline.encode_queries(
            query_tokens=query_tokens,
            query_mask=query_mask,
            raw_sequences=raw_sequences,
            use_seq=use_seq,
            use_struct=use_struct,
        )

        fused_list = []
        seq_scores_list = []
        struct_scores_list = []
        seq_indices_list = []
        struct_indices_list = []

        for b in range(query_tokens.shape[0]):
            q_tok = query_tokens[b]

            seq_fused = None
            struct_fused = None
            seq_scores = torch.zeros(self.top_k, device=q_tok.device)
            struct_scores = torch.zeros(self.top_k, device=q_tok.device)
            seq_indices = torch.full((self.top_k,), -1, device=q_tok.device, dtype=torch.long)
            struct_indices = torch.full((self.top_k,), -1, device=q_tok.device, dtype=torch.long)

            if use_seq:
                if vectors.seq is None:
                    raise RuntimeError("Sequence source enabled but query pipeline did not return seq vectors")
                if self.seq_retriever is None or self.seq_db_proj is None:
                    raise RuntimeError("Sequence source enabled but seq retriever/projection not configured")
                seq_fused, seq_scores, seq_indices = self._run_source(
                    source="seq",
                    query_tokens=q_tok,
                    query_vec=vectors.seq[b],
                    retriever=self.seq_retriever,
                    db_proj=self.seq_db_proj,
                    fusion=self.seq_fusion,
                )

            if use_struct:
                if vectors.struct is None:
                    raise RuntimeError("Structure source enabled but query pipeline did not return struct vectors")
                if self.struct_retriever is None or self.struct_db_proj is None:
                    raise RuntimeError("Structure source enabled but struct retriever/projection not configured")
                struct_fused, struct_scores, struct_indices = self._run_source(
                    source="struct",
                    query_tokens=q_tok,
                    query_vec=vectors.struct[b],
                    retriever=self.struct_retriever,
                    db_proj=self.struct_db_proj,
                    fusion=self.struct_fusion,
                )

            if seq_fused is not None and struct_fused is not None:
                mix = torch.softmax(self.source_mix_logits, dim=0)
                fused = mix[0] * seq_fused + mix[1] * struct_fused
            elif seq_fused is not None:
                fused = seq_fused
            else:
                assert struct_fused is not None
                fused = struct_fused

            fused_list.append(fused)
            seq_scores_list.append(seq_scores)
            struct_scores_list.append(struct_scores)
            seq_indices_list.append(seq_indices)
            struct_indices_list.append(struct_indices)

        fused_batch = torch.stack(fused_list, dim=0)
        model_batch = self.injection_plan.apply(tensor_batch, fused_batch)
        outputs = self.openfold(model_batch)

        outputs["seq_retrieval_scores"] = torch.stack(seq_scores_list, dim=0).detach()
        outputs["struct_retrieval_scores"] = torch.stack(struct_scores_list, dim=0).detach()
        outputs["seq_retrieval_indices"] = torch.stack(seq_indices_list, dim=0).detach()
        outputs["struct_retrieval_indices"] = torch.stack(struct_indices_list, dim=0).detach()
        outputs["retrieval_source_weights"] = torch.softmax(self.source_mix_logits, dim=0).detach()
        return outputs
