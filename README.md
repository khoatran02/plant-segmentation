# PlantSeg: SAM 3 disease segmentation

This repository fine-tunes and evaluates Hugging Face Transformers `Sam3Model` on PlantSeg. It supports single-device execution and distributed training with `torchrun`. The primary task supplies the disease name and segments the corresponding lesion; an all-classes evaluation mode also tests predictions without supplying the true disease. These are different protocols and their scores should not be compared as if they were equivalent.

## Setup

Use Python 3.12 or 3.13. Install dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For Kaggle, select **GPU T4 x2**, enable Internet for the first installation and model download, and run `bash scripts/setup_sam3.sh` (or `%pip install -r requirements.txt` in a notebook). Restart the notebook kernel after installing. The requirements pin compatible PyTorch and Transformers versions; an incompatible preinstalled `torchaudio` can cause an undefined-symbol error, in which case reinstall the requirements and restart the kernel.

Access to [facebook/sam3](https://huggingface.co/facebook/sam3) requires approval on Hugging Face. Set `HF_TOKEN` to a token with access, for example using a Kaggle Secret named `HF_TOKEN`, or authenticate with `hf auth login`. Do not print or commit the token. If authentication succeeds but model access fails, check that the account and token have access to the gated model. Model config, tokenizer, processor and base weights are downloaded to the Hugging Face cache. Adaptation checkpoints do not contain the full base weights. Use `--cache_dir` to choose the cache and `--local_files_only` for offline runs with a complete cache or local model directory.

## Dataset

Pass `--dataset_dir` pointing directly to a PlantSeg directory containing `Metadata.csv`. The default checks `dataset/plantseg` and falls back to `datasets/plantseg` when appropriate:

```text
plantseg/
  Metadata.csv
  images/{train,val,test}/*.jpg
  annotations/{train,val,test}/*.png
  annotation_{train,val,test}.json  # required only for COCO annotations
```

PNG masks are the default ground truth: 0 is background and 1–115 are disease classes; metadata indices are 0–114. Images are EXIF-transposed to align with masks. Invalid dimensions, missing pairs and invalid labels raise errors. The audited local dataset contained 5,367 train, 846 validation and 1,561 test images; counts depend on the dataset version. The audit found 12 PNG/metadata label disagreements. Use the same `--label_policy` in training and evaluation:

- `mask` (default): use the PNG label for the prompt and category.
- `metadata`: retain the foreground pixels but assign the metadata class in memory.
- `strict`: fail on disagreement.

The optional `--annotation_source coco` uses polygon annotations for training; evaluation still uses PNG masks. PNG connected components become lossless RLE pseudo-instances, not expert-verified lesion instances. If an image has more than 200 components, the smaller components are merged. Do not use these pseudo-instances to claim instance AP. Audit findings also included invalid COCO polygons and cross-split duplicate or near-duplicate images; account for potential leakage when interpreting scores.

Prepare and validate the data on CPU:

```bash
python train.py --prepare_only --dataset_dir datasets/plantseg \
  --output_dir outputs/sam3-plantseg --annotation_source png --label_policy mask
```

The annotation cache fingerprints dataset paths, metadata, file names, sizes and modification times, annotation source and label policy; it is not a content hash of every image. Use `--rebuild_annotations` to regenerate it. To plot image resolutions and ground-truth mask coverage, run:

```bash
python -m src.dataset_statistics --dataset_dir datasets/plantseg \
  --output_dir outputs/dataset-statistics
```

Add `--splits test` to restrict the analysis. Outputs include PNG/SVG plots, per-image CSV and JSON summaries.

## Train with a known disease

`--known_disease` supplies the correct disease prompt for each image and trains semantic BCE and Dice losses. Validation uses an oracle disease label and semantic masks, without query or presence filtering. This measures disease-conditioned segmentation, not disease classification. The Kaggle launcher starts a new run from adaptation weights:

```bash
DATASET_DIR=/kaggle/input/<dataset>/plantseg \
INIT_FROM=/kaggle/input/<checkpoint>/last.pt \
OUTPUT_DIR=/kaggle/working/plantseg_known_disease \
bash scripts/train_kaggle_known_disease.sh
```

Replace the paths with the mounted dataset and checkpoint. The preset uses 672 × 672 images, one image per GPU, gradient accumulation of 8, 20 epochs and validation every epoch. Select thresholds on validation, then evaluate the test set once with the selected configuration. `scripts/train_kaggle_fast.sh` provides a shorter known-disease run with baseline validation.

`INIT_FROM` / `--init_from` loads adaptation weights into a **new output directory** and resets optimizer, schedule, epoch and best metric. Use it when changing the training objective or input resolution. `RESUME` / `--resume` instead restores a compatible run at an epoch boundary; preserve the original dataset, training configuration, model identity, world size, input size and total epoch count. The two options are mutually exclusive. Legacy format 2 upstream-backend checkpoints cannot be resumed with the Transformers backend, which uses format 3. Old checkpoints without an input size are interpreted as the original model size (1008 for facebook/sam3).

## General training and all-classes evaluation

For training with positive and negative disease prompts and combined instance/semantic supervision:

```bash
DATASET_DIR=/kaggle/input/<dataset>/plantseg \
MODEL_ID=facebook/sam3 \
bash scripts/train_kaggle_2xt4.sh
```

This launcher defaults to 672 × 672, validation every 3 epochs and evaluation prompt batches of 2; the last epoch is always validated. Override `IMAGE_SIZE`, `VAL_EVERY`, `EVAL_PROMPT_BATCH_SIZE`, `OUTPUT_DIR` and other launcher environment variables as needed. `heads` freezes the vision backbone and text encoder while training the neck, projection, fusion/decoder and heads. `full` trains the whole model and requires more memory. Run a small GPU smoke test before a long job; CPU tests do not establish that the full model fits on a T4. See `python train.py --help` and the launcher for all options.

`all_classes` evaluates every disease prompt without receiving the ground-truth class, shares image features and combines predictions across classes. It is substantially slower than `oracle`, which supplies the true class and is useful for mask diagnostics. Increase prompt batching only after checking memory and metrics. On a resumed run, `--resume_allow_val_schedule_change` permits changing `val_every` and resets early-stopping patience; it does not permit changing the evaluation protocol or training hyperparameters.

The Transformers backend implements its own Hungarian-matching objective for the general protocol, rather than the upstream auxiliary/one-to-many recipe. Input resizing, processor configuration and model grids are updated together; custom facebook/sam3 image sizes must be positive multiples of 28. `--loss_mask_size` affects training loss resolution, while evaluation resizes logits to the original mask size. Record the model revision and resolved commit from `run-config.json` for reproducibility.

## Evaluate and run inference

Evaluate a known-disease checkpoint on Kaggle (the script writes metrics, predictions and a log):

```bash
DATASET_DIR=/kaggle/input/<dataset>/plantseg \
CHECKPOINT=/kaggle/working/plantseg_known_disease/best.pt \
OUTPUT_DIR=/kaggle/working/known_disease_test \
bash scripts/test_kaggle_known_disease.sh
```

Set `NPROC_PER_NODE=1` for one GPU. For local MPS or CPU evaluation:

```bash
python evaluate.py --dataset_dir datasets/plantseg --split test \
  --resume outputs/best.pt --known_disease --image_size 672 \
  --device mps --amp off --cache_dir outputs/hf-cache --local_files_only \
  --save_predictions outputs/test/predictions --output_dir outputs/test
```

Use `--device cpu` if MPS is unavailable. The image size and model identity must match the checkpoint. To evaluate without the true disease label, use `--eval_mode all_classes` with a compatible general-protocol checkpoint; omit `--resume` for a pretrained zero-shot baseline. Use `python evaluate.py --help` for the complete options.

Evaluate another model's predictions without loading SAM 3:

```bash
python evaluate.py --dataset_dir datasets/plantseg --split test \
  --prediction_dir /path/to/class-index-pngs --label_policy mask \
  --output_dir outputs/baseline-test
```

Predictions must match the ground-truth file names and original dimensions, be single-channel integer PNGs, and use 0 for background and 1–115 for diseases. The evaluator does not resize, decode RGB colors or infer a class for binary masks. Missing predictions are errors.

For reproducible visualizations of a sample of test predictions:

```bash
python inference.py --checkpoint outputs/best.pt --dataset_dir datasets/plantseg \
  --cache_dir outputs/hf-cache --local_files_only --output_dir outputs/inference
```

`inference.py` detects the known-disease protocol from newer checkpoints; see `--help` for sampling and device options.

Metrics are reported on a 0–1 scale. `miou` and `macc` pool pixels over the split first, then average disease-class IoU and pixel accuracy, excluding background. A class with zero union is omitted from mIoU; mAcc omits classes with no ground-truth pixels. Results also include background-inclusive, foreground and per-image metrics, class support, CSV reports and a pixel confusion matrix. Oracle metrics are not classification accuracy. Comparisons require matching dataset split, label policy, background/absent-class policy, preprocessing and evaluation protocol; this SAM 3 training recipe does not reproduce the baseline training recipe of the [PlantSeg paper](https://doi.org/10.1038/s41597-025-06513-4).

Training output includes `best.pt`, `last.pt`, `run-config.json`, processor files, logs and validation metrics. Checkpoints include adaptation weights and training state, not the full base model. CUDA kernels are not guaranteed to resume bitwise deterministically.

## Local demo

```bash
python demo.py --checkpoint outputs/best.pt --dataset_dir datasets/plantseg \
  --cache_dir outputs/hf-cache --device auto --port 8000 --reload
```

Open <http://127.0.0.1:8000>, wait for the model, select or upload an image, enter a prompt and run. The interface remains in Vietnamese. `--reload` restarts the server when demo source files change; omit it when unnecessary. The standard-library server selects CUDA, MPS or CPU automatically. It uses cached base weights by default; add `--allow_download` if necessary and the Hugging Face account has access.

If the checkpoint is missing or fails to load, the page allows uploading a compatible `.pt`, `.pth` or `.ckpt` file (up to 4 GB). Uploads are stored in `outputs/demo-checkpoints/` by default, without overwriting `outputs/best.pt`; set `--checkpoint_upload_dir` to change this. To use an uploaded file after restarting, pass its path via `--checkpoint`. The base model must still be available separately.

The five visible samples in `demo-samples/`, and optional images in `demo-samples-extra/`, are recognized by their EXIF-corrected RGB pixels. For recognized images the demo looks up the reference disease label and uses it as the SAM 3 prompt; the general prompt does not affect the label or mask. This lookup is not a classification prediction. For other images, the demo uses the user's prompt and an **experimental** presence/region-overlap heuristic to rank disease names. It is not a validated classifier; a known-disease checkpoint was not trained to distinguish disease names. Predicted masks are model outputs, not ground truth.

Uploads accept JPG, PNG or WebP up to 12 MB and 20 megapixels. A prompt is required, limited to 200 characters and the text encoder's token limit. The mask threshold is adjustable from 0.1 to 0.9; coverage refers to the whole image. The API exposes `GET /api/status`, `/api/samples`, `/api/sample/{id}`, `/api/samples.zip`, and `POST /api/predict`, `/api/checkpoint`. Prediction accepts `sample_id` or a base64 `image`, plus `prompt` and optional `threshold`. `classification.source` distinguishes `reference_manifest` from experimental `model` output.

## Repository and checks

- `train.py`, `evaluate.py`, `inference.py`, `demo.py`: command-line entry points and demo server.
- `src/`: dataset preparation, SAM 3 adapter, losses, training, evaluation and metrics.
- `scripts/`: installation and Kaggle launchers.
- `demo/`, `demo-samples/`: demo UI and sample images.
- `tests/`: unit, integration and distributed-training tests.

Run checks from the repository root:

```bash
python -m pytest -q
node --check demo/app.js
node tests/test_demo_ui.cjs
ruff check src tests train.py evaluate.py
ruff format --check src tests train.py evaluate.py
```

The tests exercise small local Transformers SAM 3 models and CPU/Gloo distributed execution; they do not replace a full-model CUDA/NCCL smoke test. See the [Transformers SAM 3 documentation](https://huggingface.co/docs/transformers/model_doc/sam3) for the model API.
