#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'USAGE' >&2
Usage:
  scripts/retrieval/monitor_openproteinnet_val_vanilla_openfold.sh <job_id> <output_root> [interval_sec]

Example:
  scripts/retrieval/monitor_openproteinnet_val_vanilla_openfold.sh 7524000 \
    /insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet/vanilla_openfold_val 60
USAGE
  exit 2
fi

JOB_ID="$1"
OUTPUT_ROOT="$2"
INTERVAL="${3:-60}"

PRED_DIR="${OUTPUT_ROOT}/run/predictions"
REPORT_SUMMARY="${OUTPUT_ROOT}/reports/summary.json"
TOTAL="unknown"
if [[ -f "${REPORT_SUMMARY}" ]]; then
  TOTAL="$(python - <<PY
import json
from pathlib import Path
p=Path("${REPORT_SUMMARY}")
try:
    d=json.loads(p.read_text())
    print(d.get("num_records_requested","unknown"))
except Exception:
    print("unknown")
PY
)"
fi

echo "Monitoring job ${JOB_ID}"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  interval:    ${INTERVAL}s"
echo "  total:       ${TOTAL}"

while true; do
  NOW="$(date '+%Y-%m-%d %H:%M:%S')"
  SQ="$(squeue -h -j "${JOB_ID}" -o '%i %t %M %R' || true)"
  COUNT=0
  if [[ -d "${PRED_DIR}" ]]; then
    COUNT="$(find "${PRED_DIR}" -maxdepth 1 -type f -name '*_unrelaxed.pdb' | wc -l | awk '{print $1}')"
  fi

  if [[ -n "${SQ}" ]]; then
    echo "[${NOW}] running: ${SQ} | predictions=${COUNT}/${TOTAL}"
    sleep "${INTERVAL}"
    continue
  fi

  SACCT="$(sacct -j "${JOB_ID}" --format=JobID,State,Elapsed,ExitCode -n -P 2>/dev/null | head -n 5 || true)"
  echo "[${NOW}] job not in squeue; sacct:"
  echo "${SACCT}"
  echo "[${NOW}] final predictions=${COUNT}/${TOTAL}"
  if [[ -f "${REPORT_SUMMARY}" ]]; then
    echo "[${NOW}] summary:"
    cat "${REPORT_SUMMARY}"
  else
    echo "[${NOW}] summary.json not found yet: ${REPORT_SUMMARY}"
  fi
  break
done
