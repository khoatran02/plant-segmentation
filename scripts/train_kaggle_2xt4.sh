#!/usr/bin/env bash
# DATASET_DIR=/kaggle/input/<dataset>/plantseg bash scripts/train_kaggle_2xt4.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATASET_DIR:?Set DATASET_DIR to the folder containing Metadata.csv, images/ and annotations/}"
OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/plantseg_sam3_transformers_s1105}"
MODEL_ID="${MODEL_ID:-facebook/sam3}"
MODEL_REVISION="${MODEL_REVISION:-main}"
CACHE_DIR="${CACHE_DIR:-/kaggle/working/hf-cache}"
OPTIONS=()
if [[ -n "${SAM3_CHECKPOINT:-}" ]]; then
  echo 'SAM3_CHECKPOINT is no longer used. Set MODEL_ID to a Hugging Face ID or save_pretrained directory.' >&2
  exit 2
fi
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  OPTIONS+=(--local_files_only)
fi
if [[ -n "${RESUME:-}" ]]; then
  OPTIONS+=(--resume "$RESUME")
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Detect incompatible native extensions before spending time building annotations.
python -c 'from src.sam3_adapter import load_sam3_api; load_sam3_api(); print("[setup] Transformers SAM3 imports OK.")'
# Check model authorization before the separate CPU annotation-preparation command.
python -c 'import sys; from src.training import parse_args; from src.sam3_adapter import check_model_access; check_model_access(parse_args(sys.argv[1:])); print("[setup] Model access check completed (skipped in local/offline mode).")' \
  --model_id "$MODEL_ID" --model_revision "$MODEL_REVISION" --cache_dir "$CACHE_DIR" "${OPTIONS[@]}" "$@"
python train.py --prepare_only --dataset_dir "$DATASET_DIR" --output_dir "$OUTPUT_DIR" --annotation_source png --label_policy mask
# The Transformers library downloads config, processor and safetensors into CACHE_DIR.
# HF_TOKEN is inherited by torchrun workers; do not pass tokens as command-line arguments.
# Effective batch = 2 GPUs * 1 image/GPU * 8 accumulation = 16 images/update.
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
  --dataset_dir "$DATASET_DIR" --output_dir "$OUTPUT_DIR" \
  --model_id "$MODEL_ID" --model_revision "$MODEL_REVISION" --cache_dir "$CACHE_DIR" \
  --device cuda --scope heads --annotation_source png --label_policy mask \
  --amp fp16 --amp_init_scale 1024 --frozen_dtype fp16 --gradient_checkpointing \
  --batch_size 1 --gradient_accumulation_steps 8 \
  --epochs 20 --warmup_ratio 0.05 --early_stopping_patience 4 \
  --lr 8e-5 --lr_vision 2.5e-5 --lr_language 5e-6 --weight_decay 0.1 \
  --gradient_clip 0.1 --prompts_per_image 1 --pos_prob 0.7 --loss_mask_size 144 \
  --eval_mode "${EVAL_MODE:-all_classes}" --selection_metric miou \
  --score_threshold 0.5 --mask_threshold 0.5 --val_every 1 \
  --num_workers 2 --log_every_updates 25 --seed 1105 \
  "${OPTIONS[@]}" "$@"
