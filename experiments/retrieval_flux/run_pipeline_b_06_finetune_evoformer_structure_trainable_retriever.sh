#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/submit_retrieval_matrix_job.sh" \
  --pipeline "b" \
  --run-tag "retrieval_opn_b06_evo_struct_retriever_postinput_effbs128_lr5e-5" \
  --train-flags "--train_evoformer --train_structure_module" \
  -- "$@"
