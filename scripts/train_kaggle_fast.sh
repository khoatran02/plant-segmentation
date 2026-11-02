#!/usr/bin/env bash
# Short known-disease mask experiment initialized from an existing run.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATASET_DIR:?Set DATASET_DIR to the PlantSeg folder}"
: "${INIT_FROM:?Set INIT_FROM to the existing SAM3 adaptation last.pt or best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/plantseg_sam3_fast_known_disease_s1105}"
MODEL_ID="${MODEL_ID:-facebook/sam3}"
MODEL_REVISION="${MODEL_REVISION:-main}"
CACHE_DIR="${CACHE_DIR:-/kaggle/working/hf-cache}"
IMAGE_SIZE="${IMAGE_SIZE:-672}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_UPDATES="${MAX_UPDATES:-300}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
OPTIONS=()
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  OPTIONS+=(--local_files_only)
fi
if [[ -e "$OUTPUT_DIR/last.pt" ]]; then
  echo "Output already contains last.pt; select a new OUTPUT_DIR." >&2
  exit 2
fi
if [[ ! -f "$INIT_FROM" ]]; then
  echo "Checkpoint does not exist: $INIT_FROM" >&2
  exit 2
fi
# Reuse the previous annotation cache; normal fingerprint checks still apply.
# No image/mask data or test annotations are copied.
mkdir -p "$OUTPUT_DIR/coco"
for split in train val; do
  annotation_cache="$(dirname "$INIT_FROM")/coco/${split}-png-mask.json"
  if [[ -f "$annotation_cache" && ! -e "$OUTPUT_DIR/coco/${split}-png-mask.json" ]]; then
    cp "$annotation_cache" "$OUTPUT_DIR/coco/${split}-png-mask.json"
  fi
done
if [[ "${BASELINE_EVAL:-1}" == "1" ]]; then
  torchrun --standalone --nnodes=1 --nproc_per_node=2 evaluate.py \
    --dataset_dir "$DATASET_DIR" --model_id "$MODEL_ID" --model_revision "$MODEL_REVISION" \
    --cache_dir "$CACHE_DIR" --resume "$INIT_FROM" --scope heads --image_size "$IMAGE_SIZE" \
    --split val --known_disease --eval_mode oracle --eval_mask_source semantic --amp fp16 \
    --score_threshold 0.5 --mask_threshold 0.5 \
    --output_dir "$OUTPUT_DIR/before" "${OPTIONS[@]}"
fi
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
  --dataset_dir "$DATASET_DIR" --output_dir "$OUTPUT_DIR" --init_from "$INIT_FROM" \
  --model_id "$MODEL_ID" --model_revision "$MODEL_REVISION" --cache_dir "$CACHE_DIR" \
  --scope heads --annotation_source png --label_policy mask --image_size "$IMAGE_SIZE" \
  --known_disease --loss_mode semantic --eval_mask_source semantic \
  --amp fp16 --amp_init_scale 1024 --frozen_dtype fp16 --no-gradient_checkpointing \
  --batch_size "$BATCH_SIZE" --gradient_accumulation_steps 1 \
  --epochs 3 --max_updates "$MAX_UPDATES" --warmup_ratio 0.05 \
  --lr 1e-4 --lr_vision 2.5e-5 --lr_language 5e-6 --weight_decay 0.1 \
  --gradient_clip 0.1 --prompts_per_image 1 --pos_prob 1 --loss_mask_size 144 \
  --eval_mode oracle --selection_metric miou --val_every 20 --early_stopping_patience 0 \
  --score_threshold 0.5 --mask_threshold 0.5 --eval_prompt_batch_size 1 \
  --num_workers 4 --log_every_updates 25 --seed 1105 \
  "${OPTIONS[@]}" "$@"
