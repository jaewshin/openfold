#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/submit_retrieval_matrix_job.sh" \
  --pipeline "a" \
  --run-tag "retrieval_opn_a05_input_evo_retriever_legacy_effbs128_lr5e-5" \
  --train-flags "--train_input_embedder --train_evoformer" \
  -- "$@"
