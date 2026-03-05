from __future__ import annotations

from typing import Dict, Optional, Tuple

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
        retrieval_pipeline: str = "legacy",
        seq_retriever: Optional[LazyFaissRetriever] = None,
        struct_retriever: Optional[LazyFaissRetriever] = None,
        seq_db_proj: Optional[nn.Linear] = None,
        struct_db_proj: Optional[nn.Linear] = None,
        seq_row_lookup: Optional[object] = None,
        seq_sequence_store: Optional[object] = None,
        seq_context_encoder: Optional[nn.Module] = None,
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
        self.retrieval_pipeline = str(retrieval_pipeline).strip().lower()

        self.seq_retriever = seq_retriever
        self.struct_retriever = struct_retriever
        self.seq_db_proj = seq_db_proj
        self.struct_db_proj = struct_db_proj
        self.seq_row_lookup = seq_row_lookup
        self.seq_sequence_store = seq_sequence_store
        self.seq_context_encoder = seq_context_encoder

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

    def _lookup_seq_raw_components(self):
        if self.seq_row_lookup is None or self.seq_sequence_store is None:
            raise RuntimeError(
                "rawseq_esm1b_ragstyle requires configured seq_row_lookup and seq_sequence_store."
            )
        if self.seq_context_encoder is None:
            raise RuntimeError("rawseq_esm1b_ragstyle requires configured seq_context_encoder.")
        return self.seq_row_lookup, self.seq_sequence_store, self.seq_context_encoder

    def _prepare_rawseq_seq_context(
        self,
        indices: torch.Tensor,
        scores: torch.Tensor,
        valid: torch.Tensor,
        emb_dim: int,
        device: torch.device,
    ):
        id_lookup, seq_store, context_encoder = self._lookup_seq_raw_components()

        seq_ids = []
        seqs = []
        content_valid = valid.clone()
        for j, idx in enumerate(indices.tolist()):
            if not bool(valid[j].item()):
                seq_ids.append(None)
                seqs.append(None)
                continue

            seq_id = id_lookup.get(int(idx))
            if seq_id is None:
                content_valid[j] = False
                seq_ids.append(None)
                seqs.append(None)
                continue

            seq = seq_store.get(seq_id)
            if not seq:
                content_valid[j] = False
                seq_ids.append(None)
                seqs.append(None)
                continue

            seq_ids.append(seq_id)
            seqs.append(seq)

        pairs = [
            (sid, seq)
            for sid, seq, ok in zip(seq_ids, seqs, content_valid.tolist())
            if ok and sid is not None and seq is not None
        ]
        emb_map = context_encoder.encode_with_ids(pairs) if pairs else {}

        max_len = 1
        for sid, ok in zip(seq_ids, content_valid.tolist()):
            if not ok or sid is None:
                continue
            emb = emb_map.get(sid)
            if emb is not None:
                max_len = max(max_len, int(emb.shape[0]))

        zero_row = torch.zeros(max_len, emb_dim, device=device, dtype=torch.float32)
        zero_mask = torch.zeros(max_len, device=device, dtype=torch.float32)
        retrieved_rows = []
        retrieved_masks = []

        for j, (sid, ok) in enumerate(zip(seq_ids, content_valid.tolist())):
            if not ok or sid is None:
                retrieved_rows.append(zero_row.clone())
                retrieved_masks.append(zero_mask.clone())
                continue

            emb = emb_map.get(sid)
            if emb is None:
                content_valid[j] = False
                retrieved_rows.append(zero_row.clone())
                retrieved_masks.append(zero_mask.clone())
                continue

            emb = emb.to(device=device, dtype=torch.float32)
            if int(emb.shape[-1]) != int(emb_dim):
                raise ValueError(
                    f"Retrieved raw-sequence embedding dim mismatch: got {int(emb.shape[-1])}, "
                    f"expected {int(emb_dim)}."
                )
            n = min(int(emb.shape[0]), max_len)
            emb_n = emb[:n]
            if n < max_len:
                emb_n = torch.cat(
                    [emb_n, torch.zeros(max_len - n, emb_dim, device=device, dtype=emb_n.dtype)],
                    dim=0,
                )
            mask_n = torch.cat(
                [
                    torch.ones(n, device=device, dtype=torch.float32),
                    torch.zeros(max_len - n, device=device, dtype=torch.float32),
                ],
                dim=0,
            )
            retrieved_rows.append(emb_n)
            retrieved_masks.append(mask_n)

        if content_valid.any():
            scores = scores.masked_fill(~content_valid, -1e9)
            scores = torch.softmax(scores, dim=0)
        else:
            scores = torch.zeros_like(scores)

        retrieved_tokens = torch.stack(retrieved_rows, dim=0)
        retrieved_masks_t = torch.stack(retrieved_masks, dim=0)
        stats = {
            "retrieval_valid_hits": float(content_valid.sum().item()),
            "context_tokens_used": float(retrieved_masks_t.sum().item()),
            "skip_cross_applied": 0.0,
        }
        return retrieved_tokens, retrieved_masks_t, scores, stats

    def _run_source(
        self,
        source: str,
        query_tokens: torch.Tensor,
        query_vec: torch.Tensor,
        retriever: LazyFaissRetriever,
        db_proj: Optional[nn.Linear],
        fusion: FusionStrategy,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        scores, indices, retrieved_keys, valid = search_index(
            query_vec=query_vec,
            retriever=retriever,
            top_k=self.top_k,
        )

        if self.retrieval_pipeline == "rawseq_esm1b_ragstyle" and source == "seq":
            retrieved_tokens, retrieved_masks, scores, stats = self._prepare_rawseq_seq_context(
                indices=indices,
                scores=scores,
                valid=valid,
                emb_dim=int(query_tokens.shape[-1]),
                device=query_tokens.device,
            )
            retrieved_tokens = retrieved_tokens.to(dtype=query_tokens.dtype)
        else:
            if db_proj is None:
                raise RuntimeError(f"Source={source} requires db projection for non-rawseq retrieval.")
            # Reconstructed vectors are per-hit vectors; represent each as 1 token.
            retrieved_tokens = db_proj(retrieved_keys.unsqueeze(1))
            retrieved_masks = valid.float().unsqueeze(-1)
            stats = {
                "retrieval_valid_hits": float(valid.sum().item()),
                "context_tokens_used": float(retrieved_masks.sum().item()),
                "skip_cross_applied": 0.0,
            }

        fused = fusion(
            query_tokens,
            retrieved_tokens,
            scores,
            retrieved_masks=retrieved_masks,
            context={"source": source},
        )
        fusion_stats_fn = getattr(fusion, "get_last_stats", None)
        if callable(fusion_stats_fn):
            try:
                fusion_stats = fusion_stats_fn()
                if isinstance(fusion_stats, dict):
                    stats.update({k: float(v) for k, v in fusion_stats.items()})
            except Exception:
                pass

        return fused, scores, indices, stats

    def close(self) -> None:
        for obj in (self.seq_row_lookup, self.seq_sequence_store):
            close_fn = getattr(obj, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

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
        seq_valid_hits_list = []
        struct_valid_hits_list = []
        seq_context_tokens_list = []
        struct_context_tokens_list = []
        seq_skip_cross_list = []
        struct_skip_cross_list = []

        for b in range(query_tokens.shape[0]):
            q_tok = query_tokens[b]

            seq_fused = None
            struct_fused = None
            seq_scores = torch.zeros(self.top_k, device=q_tok.device)
            struct_scores = torch.zeros(self.top_k, device=q_tok.device)
            seq_indices = torch.full((self.top_k,), -1, device=q_tok.device, dtype=torch.long)
            struct_indices = torch.full((self.top_k,), -1, device=q_tok.device, dtype=torch.long)
            seq_stats = {"retrieval_valid_hits": 0.0, "context_tokens_used": 0.0, "skip_cross_applied": 0.0}
            struct_stats = {"retrieval_valid_hits": 0.0, "context_tokens_used": 0.0, "skip_cross_applied": 0.0}

            if use_seq:
                if vectors.seq is None:
                    raise RuntimeError("Sequence source enabled but query pipeline did not return seq vectors")
                if self.seq_retriever is None:
                    raise RuntimeError("Sequence source enabled but seq retriever not configured")
                seq_fused, seq_scores, seq_indices, seq_stats = self._run_source(
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
                if self.struct_retriever is None:
                    raise RuntimeError("Structure source enabled but struct retriever not configured")
                struct_fused, struct_scores, struct_indices, struct_stats = self._run_source(
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
            seq_valid_hits_list.append(float(seq_stats.get("retrieval_valid_hits", 0.0)))
            struct_valid_hits_list.append(float(struct_stats.get("retrieval_valid_hits", 0.0)))
            seq_context_tokens_list.append(float(seq_stats.get("context_tokens_used", 0.0)))
            struct_context_tokens_list.append(float(struct_stats.get("context_tokens_used", 0.0)))
            seq_skip_cross_list.append(float(seq_stats.get("skip_cross_applied", 0.0)))
            struct_skip_cross_list.append(float(struct_stats.get("skip_cross_applied", 0.0)))

        fused_batch = torch.stack(fused_list, dim=0)
        model_batch = self.injection_plan.apply(tensor_batch, fused_batch)
        outputs = self.openfold(model_batch)

        outputs["seq_retrieval_scores"] = torch.stack(seq_scores_list, dim=0).detach()
        outputs["struct_retrieval_scores"] = torch.stack(struct_scores_list, dim=0).detach()
        outputs["seq_retrieval_indices"] = torch.stack(seq_indices_list, dim=0).detach()
        outputs["struct_retrieval_indices"] = torch.stack(struct_indices_list, dim=0).detach()
        outputs["seq_retrieval_valid_hits"] = torch.tensor(
            seq_valid_hits_list, dtype=torch.float32, device=query_tokens.device
        ).detach()
        outputs["struct_retrieval_valid_hits"] = torch.tensor(
            struct_valid_hits_list, dtype=torch.float32, device=query_tokens.device
        ).detach()
        outputs["seq_context_tokens_used"] = torch.tensor(
            seq_context_tokens_list, dtype=torch.float32, device=query_tokens.device
        ).detach()
        outputs["struct_context_tokens_used"] = torch.tensor(
            struct_context_tokens_list, dtype=torch.float32, device=query_tokens.device
        ).detach()
        outputs["seq_skip_cross_applied"] = torch.tensor(
            seq_skip_cross_list, dtype=torch.float32, device=query_tokens.device
        ).detach()
        outputs["struct_skip_cross_applied"] = torch.tensor(
            struct_skip_cross_list, dtype=torch.float32, device=query_tokens.device
        ).detach()
        outputs["retrieval_valid_hits"] = (
            outputs["seq_retrieval_valid_hits"] + outputs["struct_retrieval_valid_hits"]
        ).detach()
        outputs["retrieval_context_tokens"] = (
            outputs["seq_context_tokens_used"] + outputs["struct_context_tokens_used"]
        ).detach()
        outputs["retrieval_skip_cross_applied"] = torch.maximum(
            outputs["seq_skip_cross_applied"],
            outputs["struct_skip_cross_applied"],
        ).detach()
        outputs["retrieval_source_weights"] = torch.softmax(self.source_mix_logits, dim=0).detach()
        return outputs
