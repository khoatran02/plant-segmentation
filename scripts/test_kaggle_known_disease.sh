#!/usr/bin/env bash
# Evaluate the complete test split with the supplied disease label for each image.
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET_DIR="${DATASET_DIR:-/kaggle/input/datasets/duyhoang29/plantseg/plantseg}"
CHECKPOINT="${CHECKPOINT:-/kaggle/input/datasets/duyhoang29/plant-sam3-weight/last.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/plantseg_sam3_known_disease_test}"
MODEL_ID="${MODEL_ID:-facebook/sam3}"
MODEL_REVISION="${MODEL_REVISION:-main}"
CACHE_DIR="${CACHE_DIR:-/kaggle/working/hf-cache}"
IMAGE_SIZE="${IMAGE_SIZE:-672}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
OPTIONS=()
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  OPTIONS+=(--local_files_only)
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint does not exist: $CHECKPOINT. Set CHECKPOINT to the model to evaluate." >&2
  exit 2
fi
if [[ ! -f "$DATASET_DIR/Metadata.csv" || ! -d "$DATASET_DIR/images/test" ]]; then
  echo "Invalid dataset/test split: $DATASET_DIR" >&2
  exit 2
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
mkdir -p "$OUTPUT_DIR"
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" evaluate.py \
  --dataset_dir "$DATASET_DIR" --split test \
  --model_id "$MODEL_ID" --model_revision "$MODEL_REVISION" --cache_dir "$CACHE_DIR" \
  --resume "$CHECKPOINT" --scope heads --known_disease \
  --eval_mode oracle --eval_mask_source semantic \
  --image_size "$IMAGE_SIZE" --amp fp16 --mask_threshold "$MASK_THRESHOLD" \
  --eval_prompt_batch_size 1 --label_policy mask \
  --save_predictions "$OUTPUT_DIR/predictions" --output_dir "$OUTPUT_DIR" \
  "${OPTIONS[@]}" 2>&1 | tee "$OUTPUT_DIR/evaluation.log"

python - "$OUTPUT_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

output = Path(sys.argv[1])
result = json.loads((output / "test.json").read_text())
if result["split"] != "test" or result["limited"] or not result["known_disease"]:
    raise RuntimeError("Expected a complete known-disease test-set evaluation")
for metric in ("miou", "macc"):
    if result[metric] is None or not math.isfinite(result[metric]):
        raise RuntimeError(f"Undefined test metric: {metric}")
summary = {
    "checkpoint": result["adaptation_checkpoint"],
    "protocol": "known_disease / oracle / semantic masks without confidence filtering",
    "split": "test",
    "num_images": result["num_images"],
    "miou": result["miou"],
    "macc": result["macc"],
    "miou_percent": 100 * result["miou"],
    "macc_percent": 100 * result["macc"],
    "includes_background": False,
    "mask_threshold": result["mask_threshold"],
    "evaluation_seconds": result["evaluation_seconds"],
    "metric_policy": result["metric_policy"],
}
(output / "summary.json").write_text(
    json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
)
print(f"\nTest images: {summary['num_images']}")
print(f"mIoU (disease classes, no background): {summary['miou_percent']:.4f}%")
print(f"mAcc (disease classes, no background): {summary['macc_percent']:.4f}%")
print(f"Results saved in: {output}")
PY
