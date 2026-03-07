#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/train_pipeline.conf"
submission_script="${SCRIPT_DIR}/train_pipeline_submit.sh"

usage() {
  cat <<USAGE >&2
Usage: ${SCRIPT_DIR}/queue_train_resume_pipeline.sh -n <num_jobs> -c <config.yaml> [-r] [-q <queue>] [-t <time_limit>] [-m <minutes>]

Arguments:
  -n <num_jobs>      Number of Flux jobs to chain.
  -c <config.yaml>   Experiment YAML to run.
  -r                 Resume the first job from an existing checkpoint.
  -q <queue>         Flux queue to use. Defaults to ${default_queue}.
  -t <time_limit>    Per-job walltime. Defaults to ${default_pbatch_time} on pbatch and ${default_pdebug_time} on pdebug.
  -m <minutes>       Safety margin for trainer.max_time. Defaults to queue-specific values from train_pipeline.conf.
USAGE
  exit 1
}

abspath() {
  local path="$1"
  if [ -d "$path" ]; then
    (cd "$path" && pwd)
  else
    local parent
    parent=$(cd "$(dirname "$path")" && pwd)
    printf '%s/%s\n' "$parent" "$(basename "$path")"
  fi
}

duration_to_seconds() {
  local value="$1"
  if [[ "$value" =~ ^([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
    echo $(( 10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]} ))
    return 0
  fi
  if [[ "$value" =~ ^([0-9]+)([smhd])$ ]]; then
    local amount="${BASH_REMATCH[1]}"
    local unit="${BASH_REMATCH[2]}"
    case "$unit" in
      s) echo "$amount" ;;
      m) echo $(( amount * 60 )) ;;
      h) echo $(( amount * 3600 )) ;;
      d) echo $(( amount * 86400 )) ;;
      *) return 1 ;;
    esac
    return 0
  fi
  return 1
}

parse_num_nodes() {
  local cfg="$1"
  awk '/^trainer:/{in_trainer=1; next} in_trainer && /^[^[:space:]]/{in_trainer=0} in_trainer && /^[[:space:]]+num_nodes:/{print $2; exit}' "$cfg" | sed 's/#.*//'
}

num_jobs=""
config=""
resume_first=false
queue="$default_queue"
time_limit=""
safety_margin=""

while getopts ":n:c:rq:t:m:" opt; do
  case "$opt" in
    n) num_jobs="$OPTARG" ;;
    c) config="$OPTARG" ;;
    r) resume_first=true ;;
    q) queue="$OPTARG" ;;
    t) time_limit="$OPTARG" ;;
    m) safety_margin="$OPTARG" ;;
    *) usage ;;
  esac
done
shift $((OPTIND - 1))

if [ -z "$num_jobs" ] || [ -z "$config" ]; then
  usage
fi
if ! [[ "$num_jobs" =~ ^[0-9]+$ ]] || [ "$num_jobs" -le 0 ]; then
  echo "Error: -n must be a positive integer" >&2
  exit 1
fi

config=$(abspath "$config")
if [ ! -f "$config" ]; then
  echo "Error: Config file not found: ${config}" >&2
  exit 1
fi
if [ ! -x "$submission_script" ]; then
  echo "Error: Submission script not found: ${submission_script}" >&2
  exit 1
fi

case "$queue" in
  pdebug)
    if [ -z "$time_limit" ]; then
      time_limit="$default_pdebug_time"
    fi
    if [ -z "$safety_margin" ]; then
      safety_margin="$pdebug_safety_margin"
    fi
    max_seconds=3600
    ;;
  pbatch)
    if [ -z "$time_limit" ]; then
      time_limit="$default_pbatch_time"
    fi
    if [ -z "$safety_margin" ]; then
      safety_margin="$pbatch_safety_margin"
    fi
    max_seconds=86400
    ;;
  *)
    echo "Error: Unsupported queue '${queue}'. Expected one of {pdebug, pbatch}." >&2
    exit 1
    ;;
esac

if ! [[ "$safety_margin" =~ ^[0-9]+$ ]]; then
  echo "Error: safety margin must be an integer number of minutes" >&2
  exit 1
fi

time_limit_seconds=$(duration_to_seconds "$time_limit") || {
  echo "Error: Unsupported time limit format '${time_limit}'. Use values like 1h, 24h, 30m, or HH:MM:SS." >&2
  exit 1
}
if [ "$time_limit_seconds" -gt "$max_seconds" ]; then
  echo "Error: time limit '${time_limit}' exceeds the ${queue} maximum." >&2
  exit 1
fi

num_nodes=$(parse_num_nodes "$config")
if ! [[ "$num_nodes" =~ ^[0-9]+$ ]] || [ "$num_nodes" -le 0 ]; then
  echo "Error: Could not parse trainer.num_nodes from ${config}" >&2
  exit 1
fi

name=$(basename "$config" .yaml)
run_dir="${outdir}/${name}"
mkdir -p "$run_dir" "$flux_log_dir"
chain_manifest="${run_dir}/flux_job_chain.txt"
: > "$chain_manifest"

previous_jobid=""
for ((i = 1; i <= num_jobs; i++)); do
  flux_args=(
    batch
    "--job-name=${name}_launch"
    "--output=${flux_log_dir}/${name}_launch-${i}-{{id}}.out"
    "-N${num_nodes}"
    -q "$queue"
    -t "$time_limit"
  )

  if [ -n "$previous_jobid" ]; then
    flux_args+=("--dependency=afterany:${previous_jobid}")
  fi

  flux_args+=("$submission_script" -c "$config" -m "$safety_margin")
  if [ "$i" -eq 1 ]; then
    if [ "$resume_first" = true ]; then
      flux_args+=(-r)
    fi
  else
    flux_args+=(-r)
  fi

  jobid=$("${flux_args[@]}" 2>&1)
  if [[ "$jobid" == *"Error"* ]]; then
    echo "Error submitting job ${i}:" >&2
    echo "$jobid" >&2
    exit 1
  fi
  if [[ ! "$jobid" =~ ^[a-zA-Z0-9]+$ ]]; then
    echo "Unexpected Flux job id for job ${i}: ${jobid}" >&2
    exit 1
  fi

  printf '%d\t%s\n' "$i" "$jobid" >> "$chain_manifest"
  echo "Submitted job ${i}: ${jobid}"
  previous_jobid="$jobid"
done

echo "Recorded chain metadata at ${chain_manifest}"
