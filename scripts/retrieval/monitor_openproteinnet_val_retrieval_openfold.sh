#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'USAGE' >&2
Usage:
  scripts/retrieval/monitor_openproteinnet_val_retrieval_openfold.sh <job_id> <output_root> [interval_sec]

Example:
  scripts/retrieval/monitor_openproteinnet_val_retrieval_openfold.sh 7525000 \
    /insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet/retrieval_openfold_val 120
USAGE
  exit 2
fi

JOB_ID="$1"
OUTPUT_ROOT="$2"
INTERVAL="${3:-120}"

PRED_DIR="${OUTPUT_ROOT}/run/predictions"
PER_TARGET_JSONL="${OUTPUT_ROOT}/reports/retrieval_alignment_per_target.jsonl"
ALIGN_ROOT="${OUTPUT_ROOT}/inputs/alignments"
REPORT_SUMMARY="${OUTPUT_ROOT}/reports/summary.json"

TOTAL="$(python - <<PY
import json
from pathlib import Path
manifest=Path('/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet/retrieval_ready/val.fasta')
n=0
with manifest.open('r') as fh:
    for line in fh:
        if line.startswith('>'):
            n += 1
print(n)
PY
)"

echo "Monitoring job ${JOB_ID}"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  interval:    ${INTERVAL}s"
echo "  total:       ${TOTAL}"

while true; do
  NOW="$(date '+%Y-%m-%d %H:%M:%S')"
  SQ="$(squeue -h -j "${JOB_ID}" -o '%i %t %M %R' || true)"
  PREP_COUNT=0
  PRED_COUNT=0
  if [[ -f "${PER_TARGET_JSONL}" ]]; then
    PREP_COUNT="$(wc -l < "${PER_TARGET_JSONL}" | awk '{print $1}')"
  elif [[ -d "${ALIGN_ROOT}" ]]; then
    PREP_COUNT="$(find "${ALIGN_ROOT}" -mindepth 2 -maxdepth 2 -type f -name 'bfd_uniclust_hits.a3m' | wc -l | awk '{print $1}')"
  fi
  if [[ -d "${PRED_DIR}" ]]; then
    PRED_COUNT="$(find "${PRED_DIR}" -maxdepth 1 -type f -name '*_unrelaxed.pdb' | wc -l | awk '{print $1}')"
  fi

  if [[ -n "${SQ}" ]]; then
    echo "[${NOW}] running: ${SQ} | prepared=${PREP_COUNT}/${TOTAL} predictions=${PRED_COUNT}/${TOTAL}"
    sleep "${INTERVAL}"
    continue
  fi

  SACCT="$(sacct -j "${JOB_ID}" --format=JobID,State,Elapsed,ExitCode -n -P 2>/dev/null | head -n 5 || true)"
  echo "[${NOW}] job not in squeue; sacct:"
  echo "${SACCT}"
  echo "[${NOW}] final prepared=${PREP_COUNT}/${TOTAL} predictions=${PRED_COUNT}/${TOTAL}"
  if [[ -f "${REPORT_SUMMARY}" ]]; then
    echo "[${NOW}] summary:"
    cat "${REPORT_SUMMARY}"
  else
    echo "[${NOW}] summary.json not found yet: ${REPORT_SUMMARY}"
  fi
  break
done
