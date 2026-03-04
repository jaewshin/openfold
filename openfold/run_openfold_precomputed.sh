#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  cat <<'USAGE' >&2
Usage:
  openfold/run_openfold_precomputed.sh <fasta_dir> <template_mmcif_dir> <alignments_dir> <output_dir> [extra run_pretrained_openfold.py args...]

Environment overrides:
  OPENFOLD_PYTHON          Python executable (default: python)
  OPENFOLD_CONFIG_PRESET   Config preset (default: model_3_ptm)
  OPENFOLD_MODEL_DEVICE    Torch device (default: cuda:0)
USAGE
  exit 2
fi

FASTA_DIR="$1"
TEMPLATE_MMCIF_DIR="$2"
ALIGNMENTS_DIR="$3"
OUTPUT_DIR="$4"
shift 4

OPENFOLD_PYTHON="${OPENFOLD_PYTHON:-python}"
OPENFOLD_CONFIG_PRESET="${OPENFOLD_CONFIG_PRESET:-model_3_ptm}"
OPENFOLD_MODEL_DEVICE="${OPENFOLD_MODEL_DEVICE:-cuda:0}"

if [[ ! -d "$FASTA_DIR" ]]; then
  echo "fasta_dir does not exist: $FASTA_DIR" >&2
  exit 1
fi

if [[ ! -d "$ALIGNMENTS_DIR" ]]; then
  echo "alignments_dir does not exist: $ALIGNMENTS_DIR" >&2
  exit 1
fi

if [[ ! -d "$TEMPLATE_MMCIF_DIR" ]]; then
  echo "template_mmcif_dir does not exist: $TEMPLATE_MMCIF_DIR" >&2
  exit 1
fi

if ! find "$TEMPLATE_MMCIF_DIR" -maxdepth 2 -type f -name '*.cif' -print -quit | grep -q .; then
  echo "template_mmcif_dir has no .cif files: $TEMPLATE_MMCIF_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

exec "$OPENFOLD_PYTHON" run_pretrained_openfold.py \
  "$FASTA_DIR" \
  "$TEMPLATE_MMCIF_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --use_precomputed_alignments "$ALIGNMENTS_DIR" \
  --config_preset "$OPENFOLD_CONFIG_PRESET" \
  --model_device "$OPENFOLD_MODEL_DEVICE" \
  "$@"
