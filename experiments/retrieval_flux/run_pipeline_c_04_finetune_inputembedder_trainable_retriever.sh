#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/submit_retrieval_matrix_job.sh" \
  --pipeline "c" \
  --run-tag "retrieval_opn_c04_inputembed_retriever_postevo_effbs128_lr5e-5" \
  --train-flags "--train_input_embedder" \
  -- "$@"
