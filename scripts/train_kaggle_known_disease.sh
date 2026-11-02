#!/usr/bin/env bash
# Disease labels are inputs: positive prompts, mask-only loss, oracle validation.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATASET_DIR:?Set DATASET_DIR to the PlantSeg folder}"
if [[ -z "${INIT_FROM:-}" && -z "${RESUME:-}" ]]; then
  echo 'Set INIT_FROM to the old last.pt for a new run, or RESUME to a known-disease checkpoint.' >&2
  exit 2
fi
export IMAGE_SIZE="${IMAGE_SIZE:-672}"
export OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/plantseg_sam3_known_disease_${IMAGE_SIZE}_s1105}"
export EVAL_MODE=oracle
export VAL_EVERY="${VAL_EVERY:-1}"
export EVAL_PROMPT_BATCH_SIZE=1
bash scripts/train_kaggle_2xt4.sh --known_disease --pos_prob 1 "$@"
