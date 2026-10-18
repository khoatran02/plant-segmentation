# SAM 3 / SAM 3.1 — Exploration & Fine-Tuning Guide

> **Scope:** what SAM 3 is, how the code in [`sam3/`](../sam3/) is organized, and exactly how to fine-tune it on your **PlantSeg** dataset (v6 / v7). Saved as requested to [`docs/`](./).

*Repo inspected:* `sam3 @ facebookresearch/sam3` (commit on 2026-09-30, includes SAM 3.1 Multiplex).
*Dataset inspected:* `data/plantseg-v7/` — 5 367 train / 846 val images, polygons + boxes + masks, `category_id` ∈ [0, 114] (115 disease classes across many species).

---

## Table of Contents

1. [What SAM 3 Is](#1-what-sam-3-is)
2. [Repository Map](#2-repository-map)
3. [Model Architecture](#3-model-architecture)
4. [Checkpoints & Model Builder](#4-checkpoints--model-builder)
5. [Inference (Zero-Shot)](#5-inference-zero-shot)
6. [Training System Overview](#6-training-system-overview)
7. [Losses, Matcher, Optimizer](#7-losses-matcher-optimizer)
8. [Dataset Format (What the Trainer Expects)](#8-dataset-format-what-the-trainer-expects)
9. [Your PlantSeg Dataset — Audit](#9-your-plantseg-dataset--audit)
10. [Preparing PlantSeg for SAM 3 Training](#10-preparing-plantseg-for-sam-3-training)
11. [A Ready-to-Use Training Config for PlantSeg](#11-a-ready-to-use-training-config-for-plantseg)
12. [Running Training](#12-running-training)
13. [Fine-Tuning Strategies (Frozen vs Full, Box-only vs Mask)](#13-fine-tuning-strategies-frozen-vs-full-box-only-vs-mask)
14. [Evaluation](#14-evaluation)
15. [Monitoring, Checkpointing, Resuming](#15-monitoring-checkpointing-resuming)
16. [VRAM / Speed Tips](#16-vram--speed-tips)
17. [Common Pitfalls](#17-common-pitfalls)
18. [Reference: Key Files & Configs to Copy](#18-reference-key-files--configs-to-copy)

---

## 1. What SAM 3 Is

**SAM 3** ("Segment Anything with Concepts") is Meta's unified foundation model for **promptable detection + segmentation + tracking** in images *and* videos.

| Capability | Prompt types | Output |
|---|---|---|
| Image detection & instance segmentation | free-form **text** (`"apple black rot lesion"`), boxes, points, masks | boxes + masks + scores |
| Video detection + tracking | text (per-session), then propagation | masklets with persistent IDs |
| Interactive refinement (SAM 1/2 task) | points / boxes on any frame | updated masklets |

Key inventions over SAM 2:

* **Open-vocabulary exhaustive detection** — finds *all* instances of a concept described by a short phrase or visual exemplar, evaluated on **SA-Co** (270 K unique concepts, 50× larger than prior benchmarks).
* **Presence token** — improves discrimination between close phrases (e.g. *player in white* vs *player in red*).
* **Decoupled detector–tracker** sharing a vision encoder — less task interference, scales with data.
* **Data engine** — auto-annotated 4 M+ unique concepts (largest high-quality OV segmentation dataset).

**SAM 3.1** (2026-03-27, [`RELEASE_SAM3p1.md`](../sam3/RELEASE_SAM3p1.md)) adds **Object Multiplex** — objects are grouped into fixed-capacity buckets (default 16) and tracked jointly via shared memory. Result: ~7× speedup at 128 objects on H100, +2.1 cgF1 on YT-Temporal-1B, +2.0 on MOSEv2. Model code, configs and checkpoint are separate (`facebook/sam3.1`, `sam3.1_multiplex.pt`).

Benchmarks (from [`README.md`](../sam3/README.md)):

* Image SA-Co/Gold: **54.1 cgF1** (human 72.8), LVIS AP 48.5.
* Video SA-V test: 30.3 cgF1 / 58.0 pHOTA; YT-Temporal-1B: 50.8 / 69.9.

---

## 2. Repository Map

```
sam3/
├── pyproject.toml                 # deps; extras: [notebooks] [train,dev]
├── README.md / README_TRAIN.md / RELEASE_SAM3p1.md
├── assets/                       # model_diagram.png, SA-Co figures, BPE vocab is in sam3/assets/
├── examples/                     # 8 notebooks (see §5)
├── scripts/                     # bench_compile, measure_speed, extract_*_results
├── sam3/
│   ├── __init__.py
│   ├── model_builder.py          # ← build_sam3_image_model / build_sam3_video_model / build_sam3_multiplex_*
│   ├── assets/bpe_simple_vocab_16e6.txt.gz
│   ├── sam/                     # SAM-1 transformer blocks (prompt_encoder, mask_decoder, rope)
│   ├── model/                   # 25+ modules (see §3)
│   │   ├── vitdet.py, necks.py, vl_combiner.py
│   │   ├── encoder.py, decoder.py, text_encoder_ve.py, tokenizer_ve.py
│   │   ├── sam3_image.py, sam3_video_inference.py, sam3_tracker_base.py
│   │   ├── video_tracking_multiplex*.py, sam3_multiplex_*.py
│   │   └── utils/
│   ├── agent/                   # SAM 3 Agent (LLM + SAM 3 for complex reasoning prompts)
│   ├── train/                   # ← fine-tuning lives here
│   │   ├── train.py             # entry point (Hydra + submitit)
│   │   ├── trainer.py           # Trainer (DDP, AMP, schedulers, ckpt)
│   │   ├── configs/             # odinw13/, roboflow_v100/, eval_* / saco_* / gold_* / silver_*
│   │   ├── data/                # sam3_image_dataset.py, sam3_video_dataset.py, collator.py, coco_json_loaders.py
│   │   ├── loss/                # sam3_loss.py, loss_fns.py, matcher, mask_sampling
│   │   ├── optim/               # optimizer.py, schedulers.py
│   │   ├── transforms/          # basic_for_api, segmentation.DecodeRle, filter_query_*, point_sampling
│   │   └── utils/               # checkpoint_utils, distributed, logger, train_utils
│   ├── eval/                    # coco_eval, saco_veval, TETA/HOTA toolkits, postprocessors
│   ├── perflib/                 # CUDA kernels (NMS, IoU, CC, FA3)
│   └── visualization_utils.py
└── test/
```

---

## 3. Model Architecture

```
Input image (1008×1008 default)
        │
        ▼
ViT backbone  (ViT-H, img_size=1008, patch=14, depth 32, embed 1024)
        │  global_att_blocks=(7,15,23,31), window 24, RoPE + abs pos
        ▼
Sam3DualViTDetNeck / Sam3TriViTDetNeck  (scale_factors [4,2,1,0.5] or [4,2,1])
        │  PositionEmbeddingSine (256 feats)
        ▼
SAM3VLBackbone  (visual + VETextEncoder)
        │  VETextEncoder: SimpleTokenizer(BPE 16M) → 24-layer Transformer (width 1024, heads 16, d_model 256)
        ▼
TransformerEncoderFusion (6 layers, d_model 256, 8 heads, cross+self attn)
        │
        ▼
TransformerDecoder (6 layers, 200 queries, presence_token=True, boxRPB="log", dac=True)
        │  returns pred_logits, pred_boxes, presence scores
        ▼
Direct scoring  DotProductScoring (MLP 256→2048→256, residual + LayerNorm)
        ▼
UniversalSegmentationHead + PixelDecoder (3 upsampling stages, hidden 256)
        │  masks at 1008 resolution
        ▼
SequenceGeometryEncoder (3 layers, CXBlock + TransformerEncoderLayer) — encodes box/point prompts
```

Video path adds:

* **Tracker** — `Sam3TrackerPredictor` (SAM-2 style memory: 7 maskmem frames, RoPE attention, mask memory encoder with `SimpleMaskEncoder` + `CXBlock` fuser).
* **Association heuristics** — detection↔track matching (IoU 0.1, NMS 0.1, `new_det_thresh`, `hotstart_delay=15`, recondition every 16 frames, etc.). Tuned in `build_sam3_video_model` / `build_sam3_multiplex_video_model`.
* **Multiplex variant** — `TriHeadVisionOnly` + `MultiplexController` + `VideoTrackingDynamicMultiplex` (4-layer decoupled transformer, multiplex_count=16, batched grounding).

Parameter count: ~848 M (detector + tracker sharing encoder).

File ↔ component mapping:

| File | Role |
|---|---|
| [`sam3/model/vitdet.py`](../sam3/sam3/model/vitdet.py) | ViT |
| [`sam3/model/necks.py`](../sam3/sam3/model/necks.py) | Dual/Tri necks |
| [`sam3/model/vl_combiner.py`](../sam3/sam3/model/vl_combiner.py) | VL backbone |
| [`sam3/model/text_encoder_ve.py`](../sam3/sam3/model/text_encoder_ve.py) | VE text tower |
| [`sam3/model/encoder.py`](../sam3/sam3/model/encoder.py) + [`decoder.py`](../sam3/sam3/model/decoder.py) | Fusion encoder/decoder |
| [`sam3/model/sam3_image.py`](../sam3/sam3/model/sam3_image.py) | `Sam3Image` (image model) |
| [`sam3/model/sam3_video_inference.py`](../sam3/sam3/model/sam3_video_inference.py) | `Sam3VideoInferenceWithInstanceInteractivity` |
| [`sam3/model/video_tracking_multiplex*.py`](../sam3/sam3/model/video_tracking_multiplex.py) | Multiplex tracking |
| [`sam3/model/sampler`](../sam3/sam3/model/maskformer_segmentation.py) | Pixel decoder / mask head |

---

## 4. Checkpoints & Model Builder

### 4.1 Checkpoints

| Version | HF repo | File | Size |
|---|---|---|---|
| SAM 3 | `facebook/sam3` | `sam3.pt` + `config.json` | ~3 GB |
| SAM 3.1 | `facebook/sam3.1` | `sam3.1_multiplex.pt` + `config.json` | ~3 GB |

Gated — request access on Hugging Face, then `hf auth login`. Download is automatic via `model_builder.download_ckpt_from_hf()`.

### 4.2 Builder API ([`sam3/model_builder.py`](../sam3/sam3/model_builder.py))

```python
# Image — detection + segmentation
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
model = build_sam3_image_model(
    bpe_path=None,               # default: sam3/assets/bpe_simple_vocab_16e6.txt.gz
    device="cuda",
    eval_mode=True,
    checkpoint_path=None,        # None → auto-download from HF
    load_from_HF=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
)
processor = Sam3Processor(model)

# Video (SAM 3)
from sam3.model_builder import build_sam3_video_model, build_sam3_video_predictor
video_model = build_sam3_video_model(checkpoint_path=None, device="cuda")
predictor   = build_sam3_video_predictor()  # wraps in Sam3VideoPredictorMultiGPU

# Video Multiplex (SAM 3.1) — recommended for multi-object video
from sam3.model_builder import build_sam3_multiplex_video_model, build_sam3_multiplex_video_predictor, build_sam3_predictor
mux_model = build_sam3_multiplex_video_model(multiplex_count=16, use_fa3=True)
mux_pred  = build_sam3_multiplex_video_predictor(max_num_objects=16)
pred      = build_sam3_predictor(version="sam3.1", compile=True)  # unified entry point
```

`Sam3Processor` exposes `set_image`, `set_image_batch`, `set_text_prompt`, `add_geometric_prompt`, `reset_all_prompts`, `set_confidence_threshold`, `confidence_threshold` property.

---

## 5. Inference (Zero-Shot)

### 5.1 Prerequisites

```bash
conda create -n sam3 python=3.12 && conda activate sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[notebooks]"          # from sam3/
hf auth login                           # paste HF token with access to facebook/sam3
# optional speedups:
pip install einops ninja && pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/ronghanghu/cc_torch.git
```

### 5.2 Image — text prompt

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()          # auto-downloads sam3.pt
processor = Sam3Processor(model)

image = Image.open("data/plantseg-v7/images/train/apple_black_rot_1.jpg")
state = processor.set_image(image)
output = processor.set_text_prompt(state=state, prompt="black rot lesion on apple leaf")
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
# masks: (N, H, W) bool, boxes: (N, 4) xyxy, scores: (N,)
```

Try domain prompts like `"apple leaf with black rot"`, `"diseased region"`, `"powdery mildew on leaf"` and tune `processor.set_confidence_threshold(0.3)`.

### 5.3 Image — box / point prompts

```python
state = processor.set_image(image)
output = processor.add_geometric_prompt(state=state, box=[x0,y0,x1,y1], label=True)
# or chain text + geometry: set_text_prompt then add_geometric_prompt
```

### 5.4 Video

```python
from sam3.model_builder import build_sam3_predictor
predictor = build_sam3_predictor(version="sam3.1", compile=False)

resp = predictor.handle_request({"type": "start_session", "resource_path": "/path/to/frames_or.mp4"})
sid = resp["session_id"]
predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": "black rot lesion"})
for out in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
    masks = out["out_binary_masks"]   # dict: obj_id -> (T, H, W)
```

### 5.5 Batched image inference

See [`examples/sam3_image_batched_inference.ipynb`](../sam3/examples/sam3_image_batched_inference.ipynb) — uses `processor.set_image_batch(images)`.

### 5.6 Agent (complex language prompts)

[`examples/sam3_agent.ipynb`](../sam3/examples/sam3_agent.ipynb) + [`sam3/agent/`](../sam3/sam3/agent/) — LLM decomposes prompts like *"the apple that is not affected by rust"* then calls SAM 3 iteratively. Useful when zero-shot text prompts are ambiguous.

### 5.7 Example notebooks

| Notebook | What it shows |
|---|---|
| `sam3_image_predictor_example.ipynb` | text + box prompts on images |
| `sam3_video_predictor_example.ipynb` | text prompts + point refinement on video |
| `sam3_image_batched_inference.ipynb` | batched inference |
| `sam3_agent.ipynb` | agent for compositional prompts |
| `sam3.1_video_predictor_example.ipynb` | multiplex video tracking |
| `sam3_for_sam1_task_example.ipynb` | interactive (SAM-1) mode |
| `saco_gold_silver_*` | SA-Co eval visualisation |

---

## 6. Training System Overview

### 6.1 Stack

* **Entry:** [`sam3/train/train.py`](../sam3/sam3/train/train.py) — Hydra `compose(config_name=...)`, handles local vs SLURM (`submitit`), job arrays, port sampling, `experiment_log_dir`.
* **Trainer:** [`sam3/train/trainer.py`](../sam3/sam3/train/trainer.py) — class `Trainer`. DDP (`DistributedDataParallel`), AMP (`bfloat16` by default), gradient accumulation, `InverseSquareRoot` schedulers, `PredictionDumper` meters, safe checkpointing (`.tmp` then `mv`).
* **Configs:** Hydra YAML under [`sam3/train/configs/`](../sam3/sam3/train/configs/) — `roboflow_v100/`, `odinw13/`, plus `eval_base.yaml` and `gold/silver/saco` eval configs.

### 6.2 Config anatomy

Every training config has four top-level blocks:

```yaml
paths:        # bpe_path, dataset_root, experiment_log_dir
scratch:      # all hyper-parameters (resolution 1008, d_model 256, matcher, LR, etc.)
roboflow_train / odinw_train:  # per-dataset: transforms, loss, dataset paths
trainer:      # Trainer(...) kwargs — data/train+val, model, loss, meters, optim, checkpoint, logging
launcher:     # num_nodes, gpus_per_node, experiment_log_dir
submitit:     # use_cluster, cpus_per_task, partition, port_range, job_array
```

Resolution of `${...}` is via Hydra/OmegaConf. CLI can override any field, e.g. `--num-gpus 1`.

### 6.3 Key reference configs

| Config | Use |
|---|---|
| [`roboflow_v100/roboflow_v100_full_ft_100_images.yaml`](../sam3/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml) | **Best template for PlantSeg** — 100-image full fine-tune, box or mask, few-shot friendly, 20 epochs, batch 1 |
| [`roboflow_v100/roboflow_v100_eval.yaml`](../sam3/sam3/train/configs/roboflow_v100/roboflow_v100_eval.yaml) | zero-shot eval on RF100 |
| [`odinw13/odinw_text_only_train.yaml`](../sam3/sam3/train/configs/odinw13/odinw_text_only_train.yaml) | 10-shot ODinW, good example with `all_odinw_supercategories` + `odinw35_prompts` |
| [`odinw13/odinw_text_only.yaml`](../sam3/sam3/train/configs/odinw13/odinw_text_only.yaml) | eval variant |

Study `roboflow_v100_full_ft_100_images.yaml` first — it is the simplest end-to-end example and the one this guide adapts.

---

## 7. Losses, Matcher, Optimizer

### 7.1 Matcher

`BinaryHungarianMatcherV2` (or `BinaryFocalHungarianMatcher` via `focal=True`) — cost = `2.0*cls + 5.0*L1 + 2.0*GIoU`, `alpha=0.25, gamma=2`. For O2M auxiliary: `BinaryOneToManyMatcher(topk=4, threshold=0.4, alpha=0.3)` with `o2m_weight=2.0`.

### 7.2 Losses ([`sam3/train/loss/`](../sam3/sam3/train/loss/))

`Sam3LossWrapper` wraps:

* `Boxes` — `loss_bbox: 5.0`, `loss_giou: 2.0`
* `IABCEMdetr` — `loss_ce: 20.0`, `presence_loss: 20.0`, `pos_weight: 10.0`, `alpha 0.25 gamma 2`, `use_presence=True`, `pad_n_queries 200`
* `Masks` *(only if `enable_segmentation=True`)* — `loss_mask: 200.0`, `loss_dice: 10.0`
* optional `SemanticSegCriterion`

Box-only training sets `enable_segmentation: False` and omits `Masks`. The commented block in `roboflow_v100_full_ft_100_images.yaml:107-154` shows the mask version — uncomment to enable.

Normalization: `global` (all-reduce), `scale_by_find_batch_size: True`.

### 7.3 Optimizer ([`sam3/train/optim/optimizer.py`](../sam3/sam3/train/optim/optimizer.py), [`schedulers.py`](../sam3/sam3/train/optim/schedulers.py))

* Base: `AdamW`
* Three param groups with **separate LR schedules** (`InverseSquareRootParamScheduler`):

| Group | LR (base) | Scaled by `lr_scale` |
|---|---|---|
| transformer + heads | `8e-4` | `×0.1 → 8e-5` |
| vision backbone | `2.5e-4` | `×2.5e-5` |
| language backbone | `5e-5` | `×5e-6` |

* Layer decay: `lrd=0.9` on `backbone.vision_backbone.trunk`, `pos_embed` excluded.
* Weight decay: `0.1`, except `bias` + `LayerNorm` → `0.0` (via `fvcore` `ConstantParamScheduler`).
* Gradient clip: `max_norm 0.1`.
* AMP: `bfloat16`.
* Scheduler timing: `timescale 20, warmup 20, cooldown 20` (in epochs, via `where = epoch/max_epochs`).

---

## 8. Dataset Format (What the Trainer Expects)

The trainer uses **Hydra-instantiated** datasets:

```yaml
trainer:
  data:
    train:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        img_folder: /abs/path/to/train/
        ann_file:   /abs/path/to/train/_annotations.coco.json
        load_segmentation: False   # True → also read masks
        transforms: ${...}
      batch_size: 1
      num_workers: 10
      collate_fn: ${scratch.collate_fn}
    val: { similar, plus coco_json_loader }
```

* **Annotation file:** COCO JSON (`images`, `annotations`, `categories`). Each annotation needs `image_id, category_id, bbox [x,y,w,h], segmentation` (polygon list or RLE), `area, iscrowd`. Boxes are `xywh` in absolute pixels.
* **Images:** `img_folder` + `file_name` from JSON.
* **Categories:** used to build **text queries**. loader `COCO_FROM_JSON` ([`sam3/train/data/coco_json_loaders.py`](../sam3/sam3/train/data/coco_json_loaders.py)) groups anns per image and creates one `FindQuery` per category phrase. `prompts` maps `category_id → phrase`; if `null`, category `name` is used directly.
* **Negatives & chunks:** `include_negatives`, `category_chunk_size` control how many negative phrases are sampled per image.
* **Transforms pipeline:** `ComposeAPI` → `FilterCrowds` → `RandomizeInputBbox` → `DecodeRle` → `RandomResizeAPI` → `PadToSizeAPI` → `ToTensorAPI` → `FilterEmptyTargets` → `NormalizeAPI`. Resolution `1008`, mean/std `0.5`.
* **Collator:** `collate_fn_api` with `with_seg_masks` matching `load_segmentation`.

COCO JSON loader does **reindexing** to contiguous IDs (`sam3/eval/coco_reindex.py:reindex_coco_to_temp`); keep `category.id` dense (0..N-1) to avoid confusion.

---

## 9. Your PlantSeg Dataset — Audit

```
data/plantseg-v7/
├── images/{train,val,test}/     # ~5 367 / 846 / ~846 images (JPG)
├── annotations/{train,val,test}/  # per-image PNG masks (not used by trainer directly)
├── annotation_{train,val,test}.json  # COCO JSON — the trainer input
└── Metadata.csv                  # human-readable index (Plant, Disease, mask ratio, URL)
```

* `annotation_train.json`: 5 367 images, 52 898 annotations, **115 `category_id` values** (0–114) even though `categories` array is empty (`[]`). Inspector found `Counter` with 115 distinct ids — each is a Plant×Disease combination. Example `bbox` is `xywh`; `segmentation` is a polygon list.
* Same structure for val/test.

> **Important:** the JSON's `categories: []` is incomplete. The trainer needs a non-empty `categories` list to build text prompts. You will regenerate it (next section). Also verify that every image file referenced by `file_name` exists under `images/train/` etc. — in this release they do.

Suggested text prompts for this domain: category names like `"apple black rot lesion"`, `"tomato late blight lesion"`, or simply `"lesion"` / `"diseased region"` if you want class-agnostic segmentation. For fine-grained evaluation keep one phrase per disease.

---

## 10. Preparing PlantSeg for SAM 3 Training

### 10.1 Generate a proper COCO categories list

Run once per split:

```python
# scripts/prepare_plantseg_coco.py
import json, pathlib

ROOT = pathlib.Path("data/plantseg-v7")
# Derive category names from Metadata.csv or from existing category_id usage.
# Option A — read Metadata.csv (Plant + Disease):
import csv
cat_names = {}  # id -> name
# Build mapping id -> "plant disease" from Metadata.csv + annotation JSONs
# If you already have a fixed taxonomy, hard-code it here.
# Example fallback: enumerate sorted unique category_ids and name them "class_{id}"
# Better: parse the per-image PNG names or Metadata.csv Disease column.

# Quick auto-recovery if categories is empty but ids exist:
for split in ["train","val","test"]:
    p = ROOT / f"annotation_{split}.json"
    coco = json.loads(p.read_text())
    ids = sorted({a["category_id"] for a in coco["annotations"]})
    if not coco.get("categories"):
        # Try to map via Metadata.csv: collect Plant+Disease for that split
        # Fallback to generic names if mapping unavailable
        coco["categories"] = [
            {"id": i, "name": f"class_{i}", "supercategory": "plant-disease"}
            for i in ids
        ]
        # --- replace with real names when you have them, e.g. ---
        # coco["categories"] = [
        #     {"id": 0, "name": "apple black rot lesion", "supercategory": "apple"},
        #     {"id": 1, "name": "apple mosaic virus lesion", "supercategory": "apple"},
        #     ...
        # ]
        (ROOT / f"annotation_{split}.with_cats.json").write_text(json.dumps(coco))
        print(f"Wrote {ROOT / f'annotation_{split}.with_cats.json'} with {len(ids)} cats")
```

Then point your training config's `ann_file` to `annotation_train.with_cats.json` / `annotation_val.with_cats.json`. Keep originals untouched.

> **Tip:** If you prefer a single prompt `"lesion"` for all diseases (class-agnostic), you can collapse categories to one id. But then you lose per-disease AP; keep 115 ids for diagnostics and merge at eval if needed.

### 10.2 Verify image paths

```bash
python -c "
import json, pathlib
for split in ['train','val']:
    coco=json.load(open(f'data/plantseg-v7/annotation_{split}.with_cats.json'))
    root=pathlib.Path(f'data/plantseg-v7/images/{split}')
    missing=[im['file_name'] for im in coco['images'] if not (root/im['file_name']).exists()]
    print(split, 'missing', len(missing), missing[:3])
"
```

The provided `file_name` values are bare filenames (e.g. `apple_black_rot_1.jpg`) and `img_folder` should be `data/plantseg-v7/images/train/` (not a per-category subfolder as in RF100/ODinW).

### 10.3 Optional: pre-check masks decode

The `DecodeRle` transform expects polygons or RLE. Your JSON uses polygons — correct. No conversion needed unless you switch to the `annotations/*.png` masks (those would require building RLE yourself; not needed).

---

## 11. A Ready-to-Use Training Config for PlantSeg

Copy the RF100 template and adapt paths. Create `sam3/train/configs/plantseg/plantseg_full_ft.yaml`:

```yaml
# @package _global_
defaults:
  - _self_

paths:
  plantseg_root: /home/workspace/works/projects/school/Master/TGMTNC/plant-segmentation/data/plantseg-v7
  experiment_log_dir: /home/workspace/works/projects/school/Master/TGMTNC/plant-segmentation/sam3_logs/plantseg
  bpe_path: /home/workspace/works/projects/school/Master/TGMTNC/plant-segmentation/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz

plantseg_train:
  num_images: null          # null = use all 5367; set 100 for few-shot ablations
  train_transforms:
    - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
      transforms:
        - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
          query_filter:
            _target_: sam3.train.transforms.filter_query_transforms.FilterCrowds
        - _target_: sam3.train.transforms.point_sampling.RandomizeInputBbox
          box_noise_std: 0.1
          box_noise_max: 20
        - _target_: sam3.train.transforms.segmentation.DecodeRle
        - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
          sizes:
            _target_: sam3.train.transforms.basic.get_random_resize_scales
            size: ${scratch.resolution}
            min_size: 480
            rounded: false
          max_size:
            _target_: sam3.train.transforms.basic.get_random_resize_max_size
            size: ${scratch.resolution}
          square: true
          consistent_transform: ${scratch.consistent_transform}
        - _target_: sam3.train.transforms.basic_for_api.PadToSizeAPI
          size: ${scratch.resolution}
          consistent_transform: ${scratch.consistent_transform}
        - _target_: sam3.train.transforms.basic_for_api.ToTensorAPI
        - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
          query_filter:
            _target_: sam3.train.transforms.filter_query_transforms.FilterEmptyTargets
        - _target_: sam3.train.transforms.basic_for_api.NormalizeAPI
          mean: ${scratch.train_norm_mean}
          std: ${scratch.train_norm_std}
        - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
          query_filter:
            _target_: sam3.train.transforms.filter_query_transforms.FilterEmptyTargets
    - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
      query_filter:
        _target_: sam3.train.transforms.filter_query_transforms.FilterFindQueriesWithTooManyOut
        max_num_objects: ${scratch.max_ann_per_img}

  val_transforms:
    - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
      transforms:
        - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
          sizes: ${scratch.resolution}
          max_size:
            _target_: sam3.train.transforms.basic.get_random_resize_max_size
            size: ${scratch.resolution}
          square: true
          consistent_transform: False
        - _target_: sam3.train.transforms.basic_for_api.ToTensorAPI
        - _target_: sam3.train.transforms.basic_for_api.NormalizeAPI
          mean: ${scratch.train_norm_mean}
          std: ${scratch.train_norm_std}

  # ── Box-only loss (fast, enough for detection). Uncomment Masks block to add segmentation. ──
  loss:
    _target_: sam3.train.loss.sam3_loss.Sam3LossWrapper
    matcher: ${scratch.matcher}
    o2m_weight: 2.0
    o2m_matcher:
      _target_: sam3.train.matcher.BinaryOneToManyMatcher
      alpha: 0.3
      threshold: 0.4
      topk: 4
    use_o2m_matcher_on_o2m_aux: false
    loss_fns_find:
      - _target_: sam3.train.loss.loss_fns.Boxes
        weight_dict: { loss_bbox: 5.0, loss_giou: 2.0 }
      - _target_: sam3.train.loss.loss_fns.IABCEMdetr
        weak_loss: False
        weight_dict: { loss_ce: 20.0, presence_loss: 20.0 }
        pos_weight: 10.0
        alpha: 0.25
        gamma: 2
        use_presence: True
        pos_focal: false
        pad_n_queries: 200
        pad_scale_pos: 1.0
    loss_fn_semantic_seg: null
    scale_by_find_batch_size: ${scratch.scale_by_find_batch_size}

scratch:
  enable_segmentation: False      # true → set collator with_seg_masks true + use Masks loss below
  d_model: 256
  pos_embed:
    _target_: sam3.model.position_encoding.PositionEmbeddingSine
    num_pos_feats: ${scratch.d_model}
    normalize: true
    scale: null
    temperature: 10000
  use_presence_eval: True
  original_box_postprocessor:
    _target_: sam3.eval.postprocessors.PostProcessImage
    max_dets_per_img: -1
    use_original_ids: true
    use_original_sizes_box: true
    use_presence: ${scratch.use_presence_eval}
  matcher:
    _target_: sam3.train.matcher.BinaryHungarianMatcherV2
    focal: true
    cost_class: 2.0
    cost_bbox: 5.0
    cost_giou: 2.0
    alpha: 0.25
    gamma: 2
    stable: False
  scale_by_find_batch_size: True
  resolution: 1008
  consistent_transform: False
  max_ann_per_img: 200
  train_norm_mean: [0.5, 0.5, 0.5]
  train_norm_std: [0.5, 0.5, 0.5]
  val_norm_mean: [0.5, 0.5, 0.5]
  val_norm_std: [0.5, 0.5, 0.5]
  num_train_workers: 8
  num_val_workers: 2
  max_data_epochs: 20
  target_epoch_size: 1500
  hybrid_repeats: 1
  context_length: 2
  gather_pred_via_filesys: false
  lr_scale: 0.1
  lr_transformer: ${times:8e-4,${scratch.lr_scale}}       # 8e-5
  lr_vision_backbone: ${times:2.5e-4,${scratch.lr_scale}} # 2.5e-5
  lr_language_backbone: ${times:5e-5,${scratch.lr_scale}} # 5e-6
  lrd_vision_backbone: 0.9
  wd: 0.1
  scheduler_timescale: 20
  scheduler_warmup: 20
  scheduler_cooldown: 20
  val_batch_size: 2
  collate_fn_val:
    _target_: sam3.train.data.collator.collate_fn_api
    _partial_: true
    repeats: ${scratch.hybrid_repeats}
    dict_key: plantseg
    with_seg_masks: ${scratch.enable_segmentation}
  gradient_accumulation_steps: 1
  train_batch_size: 2
  collate_fn:
    _target_: sam3.train.data.collator.collate_fn_api
    _partial_: true
    repeats: ${scratch.hybrid_repeats}
    dict_key: all
    with_seg_masks: ${scratch.enable_segmentation}

trainer:
  _target_: sam3.train.trainer.Trainer
  skip_saving_ckpts: false
  empty_gpu_mem_cache_after_eval: True
  skip_first_val: true
  max_epochs: ${scratch.max_data_epochs}
  accelerator: cuda
  seed_value: 123
  val_epoch_freq: 5
  mode: train
  gradient_accumulation_steps: ${scratch.gradient_accumulation_steps}
  distributed: { backend: nccl, find_unused_parameters: true, gradient_as_bucket_view: true }
  loss:
    all: ${plantseg_train.loss}
    default: { _target_: sam3.train.loss.sam3_loss.DummyLoss }
  data:
    train:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        limit_ids: ${plantseg_train.num_images}
        transforms: ${plantseg_train.train_transforms}
        load_segmentation: ${scratch.enable_segmentation}
        max_ann_per_img: 500000
        multiplier: 1
        max_train_queries: 50000
        max_val_queries: 50000
        training: true
        use_caching: False
        img_folder: ${paths.plantseg_root}/images/train/
        ann_file: ${paths.plantseg_root}/annotation_train.with_cats.json
        coco_json_loader:
          _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
          include_negatives: true
          category_chunk_size: 2
          _partial_: true
      shuffle: True
      batch_size: ${scratch.train_batch_size}
      num_workers: ${scratch.num_train_workers}
      pin_memory: True
      drop_last: True
      collate_fn: ${scratch.collate_fn}
    val:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        load_segmentation: ${scratch.enable_segmentation}
        coco_json_loader:
          _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
          include_negatives: true
          category_chunk_size: 4
          _partial_: true
        img_folder: ${paths.plantseg_root}/images/val/
        ann_file: ${paths.plantseg_root}/annotation_val.with_cats.json
        transforms: ${plantseg_train.val_transforms}
        max_ann_per_img: 100000
        multiplier: 1
        training: false
      shuffle: False
      batch_size: ${scratch.val_batch_size}
      num_workers: ${scratch.num_val_workers}
      pin_memory: True
      drop_last: False
      collate_fn: ${scratch.collate_fn_val}
  model:
    _target_: sam3.model_builder.build_sam3_image_model
    bpe_path: ${paths.bpe_path}
    device: cpu
    eval_mode: false
    enable_segmentation: ${scratch.enable_segmentation}
  meters:
    val:
      plantseg:
        detection:
          _target_: sam3.eval.coco_writer.PredictionDumper
          iou_type: bbox
          dump_dir: ${launcher.experiment_log_dir}/dumps/plantseg
          merge_predictions: True
          postprocessor: ${scratch.original_box_postprocessor}
          gather_pred_via_filesys: ${scratch.gather_pred_via_filesys}
          maxdets: 100
          pred_file_evaluators:
            - _target_: sam3.eval.coco_eval_offline.CocoEvaluatorOfflineWithPredFileEvaluators
              gt_path: ${paths.plantseg_root}/annotation_val.with_cats.json
              tide: False
              iou_type: bbox
  optim:
    amp: { enabled: true, amp_dtype: bfloat16 }
    optimizer: { _target_: torch.optim.AdamW }
    gradient_clip: { _target_: sam3.train.optim.optimizer.GradientClipper, max_norm: 0.1, norm_type: 2 }
    param_group_modifiers:
      - _target_: sam3.train.optim.optimizer.layer_decay_param_modifier
        _partial_: True
        layer_decay_value: ${scratch.lrd_vision_backbone}
        apply_to: 'backbone.vision_backbone.trunk'
        overrides: [{ pattern: '*pos_embed*', value: 1.0 }]
    options:
      lr:
        - scheduler: { _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler, base_lr: ${scratch.lr_transformer}, timescale: ${scratch.scheduler_timescale}, warmup_steps: ${scratch.scheduler_warmup}, cooldown_steps: ${scratch.scheduler_cooldown} }
        - scheduler: { _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler, base_lr: ${scratch.lr_vision_backbone}, timescale: ${scratch.scheduler_timescale}, warmup_steps: ${scratch.scheduler_warmup}, cooldown_steps: ${scratch.scheduler_cooldown} }
          param_names: ['backbone.vision_backbone.*']
        - scheduler: { _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler, base_lr: ${scratch.lr_language_backbone}, timescale: ${scratch.scheduler_timescale}, warmup_steps: ${scratch.scheduler_warmup}, cooldown_steps: ${scratch.scheduler_cooldown} }
          param_names: ['backbone.language_backbone.*']
      weight_decay:
        - scheduler: { _target_: fvcore.common.param_scheduler.ConstantParamScheduler, value: ${scratch.wd} }
        - scheduler: { _target_: fvcore.common.param_scheduler.ConstantParamScheduler, value: 0.0 }
          param_names: ['*bias*']
          module_cls_names: ['torch.nn.LayerNorm']
  checkpoint:
    save_dir: ${launcher.experiment_log_dir}/checkpoints
    save_freq: 5
    save_best_meters: []
    skip_saving_parameters: []
    model_weight_initializer:
      _target_: sam3.train.utils.checkpoint_utils.load_state_dict_into_model
      checkpoint_path: null   # ← set to sam3.pt to warm-start from pretrained
      strict: false
  logging:
    tensorboard_writer: { _target_: sam3.train.utils.logger.make_tensorboard_logger, log_dir: ${launcher.experiment_log_dir}/tensorboard, flush_secs: 120, should_log: true }
    wandb_writer: null
    log_dir: ${launcher.experiment_log_dir}/logs
    log_freq: 20
    log_scalar_frequency: 20

launcher:
  num_nodes: 1
  gpus_per_node: 1
  experiment_log_dir: ${paths.experiment_log_dir}
  multiprocessing_context: forkserver

submitit:
  use_cluster: false
  timeout_hour: 72
  cpus_per_task: 8
  port_range: [10000, 65000]
```

**To enable mask training**, flip `scratch.enable_segmentation: true` and add the `Masks` loss entry:

```yaml
# inside plantseg_train.loss.loss_fns_find, append:
      - _target_: sam3.train.loss.loss_fns.Masks
        focal_alpha: 0.25
        focal_gamma: 2.0
        weight_dict: { loss_mask: 200.0, loss_dice: 10.0 }
        compute_aux: false
```

and set both `collate_fn.with_seg_masks: true` (already via `scratch.enable_segmentation`). Add `loss_fn_semantic_seg` if you also want semantic supervision (rarely needed).

---

## 12. Running Training

### 12.1 Install training extras

```bash
cd sam3
pip install -e ".[train]"   # hydra-core, submitit, tensorboard, zstandard, scipy, torchmetrics, fvcore, fairscale, scikit-image
```

### 12.2 Warm-start from pretrained (strongly recommended)

Pretraining on SA-Co is extremely valuable. Do **not** train from scratch unless you have 100K+ images.

```bash
# download once (also auto-downloaded by model_builder, but explicit path is clearer for training)
python -c "from sam3.model_builder import download_ckpt_from_hf; print(download_ckpt_from_hf('sam3'))"
# → /home/you/.cache/huggingface/hub/.../sam3.pt
```

Then in the YAML set:

```yaml
trainer:
  checkpoint:
    model_weight_initializer:
      _target_: sam3.train.utils.checkpoint_utils.load_state_dict_into_model
      checkpoint_path: /home/you/.cache/huggingface/hub/.../sam3.pt
      strict: false
```

And `skip_saving_parameters: []` unless you freeze (see §13).

### 12.3 Launch — single GPU

```bash
# from sam3/ directory
python sam3/train/train.py \
  -c configs/plantseg/plantseg_full_ft.yaml \
  --use-cluster 0 --num-gpus 1
```

### 12.4 Launch — multi-GPU single node

```bash
python sam3/train/train.py \
  -c configs/plantseg/plantseg_full_ft.yaml \
  --use-cluster 0 --num-gpus 4
```

The launcher uses `torch.multiprocessing.spawn(start_method="spawn")` — one rank per GPU.

### 12.5 Launch — SLURM cluster

```bash
python sam3/train/train.py \
  -c configs/plantseg/plantseg_full_ft.yaml \
  --use-cluster 1 --partition gpu --account $ACCOUNT --qos normal --num-gpus 8 --num-nodes 2
```

Job arrays (sweeping many RF100 categories) use `submitit.job_array: {num_tasks: 100, task_index: 0}` — not needed for PlantSeg (single category set), so leave `use_cluster: false` or omit `job_array`.

### 12.6 What happens at launch

1. Config is composed, printed, and saved to `experiment_log_dir/{config,config_resolved}.yaml`.
2. `Trainer` is instantiated, DDP backend `nccl` is initialised, seeds set.
3. Training loops `max_epochs` (20), validating every `val_epoch_freq` (5).
4. Logs go to `experiment_log_dir/{tensorboard,logs,checkpoints,submitit_logs}`.

---

## 13. Fine-Tuning Strategies (Frozen vs Full, Box-only vs Mask)

### 13.1 Two dimensions

| Dimension | Options | When |
|---|---|---|
| **Frozen towers** | full FT vs freeze vision vs freeze language vs freeze both | small data → freeze vision; domain shift large → full FT with lower LR |
| **Segmentation** | box-only (`enable_segmentation: false`) vs box+mask (`true`) | start box-only for speed; add masks once boxes converge |

### 13.2 Freezing patterns

Via `skip_saving_parameters` + Hydra param groups (the trainer asserts frozen params are not saved):

```yaml
# Freeze vision tower (keep SA-Co visual features):
trainer:
  checkpoint:
    skip_saving_parameters: ['backbone.vision_backbone.*']
    # also lower its LR or remove its scheduler group if you want truly frozen

# Freeze language tower (keep text embeddings fixed):
  checkpoint:
    skip_saving_parameters: ['backbone.language_backbone.*']

# Freeze both towers, fine-tune only decoder + heads (parameter-efficient):
  checkpoint:
    skip_saving_parameters: ['backbone.*']
```

Alternatively use `OptimConf.param_group_modifiers` to set per-group LR to 0, or pass `--freeze_text_tower / --freeze_image_tower` if your config exposes `scratch.freeze_*` (the ODinW config does: `freeze_text_tower: False, freeze_image_tower: NoFreeze`).

**Recommendation for PlantSeg (5K images, 115 classes, lesion appearance differs from SA-Co):**

1. **Stage 1 — box-only, vision frozen** (`skip_saving_parameters: ['backbone.vision_backbone.*']`, `lr_scale 0.1`, 10 epochs) — fast convergence, sanity check.
2. **Stage 2 — unfreeze all, lower LR** (`lr_scale 0.05`, `enable_segmentation: true` with `Masks` loss, 10 more epochs) — refine masks.
3. If overfitting, increase `num_images` filtering is not needed (you have 5K > 100-shot RF100), but you can reduce `lrd_vision_backbone` or add stronger augmentation (`RandomResize` min 480 already covers scale jitter).

### 13.3 Batch / LR scaling

Configs use `train_batch_size: 1–2` per GPU with `gradient_accumulation_steps: 1`. For larger effective batch, increase `gradient_accumulation_steps` rather than `batch_size` (avoids OOM at 1008 resolution). LR schedulers are step-based via `where = epoch/max_epochs`; they auto-adapt to batch size only through `scale_by_find_batch_size`.

---

## 14. Evaluation

### 14.1 During training

`Trainer` runs `val_epoch` every `val_epoch_freq` epochs and logs meters. Meter `plantseg/detection` uses `PredictionDumper` → `CocoEvaluatorOfflineWithPredFileEvaluators` (bbox AP, with `maxdets 100`). Masks add a second dumper with `iou_type: segm`.

### 14.2 Offline / zero-shot eval

Set `trainer.mode: val` and reuse the same config:

```bash
python sam3/train/train.py -c configs/plantseg/plantseg_full_ft.yaml --use-cluster 0 --num-gpus 1
# with trainer.mode: val  (edit YAML or override via Hydra if exposed)
```

Or use the standalone evaluators in [`sam3/eval/`](../sam3/sam3/eval/):

* `coco_eval.py` / `coco_eval_offline.py` — COCO AP / cgF1.
* `cgf1_eval.py` — SA-Co's class-generic F1 (more informative for OV).
* `saco_veval_eval.py` — video VEval.
* `ytvis_eval.py` — YTVIS-style.

Postprocessor is `PostProcessImage(use_presence=True)` for joint box+presence scoring; set `max_dets_per_img: -1` for exhaustive detection (SA-Co) vs `100` for COCO.

### 14.3 Recommended metrics for PlantSeg

* **COCO AP** (bbox and segm) — overall and per-disease.
* **cgF1** — if you evaluate class-agnostic ("lesion") or want SA-Co-style scoring.
* **Mask IoU / Dice** — lesion boundary quality (critical for phytopathology).

---

## 15. Monitoring, Checkpointing, Resuming

### 15.1 Output layout

```
sam3_logs/plantseg/
├── config.yaml / config_resolved.yaml
├── checkpoints/
│   ├── checkpoint.pt          # last
│   ├── checkpoint_5.pt        # every save_freq
│   └── checkpoint_10.pt
├── tensorboard/               # → tensorboard --logdir sam3_logs/plantseg/tensorboard
├── logs/{train, val}_stats.json
├── dumps/plantseg/            # predictions for eval
└── submitit_logs/             # if cluster
```

### 15.2 TensorBoard

```bash
tensorboard --logdir sam3_logs/plantseg/tensorboard --port 6006
```

Logged: `Losses/train_all_loss`, `Losses/val_*`, `Step_Stats/*/Batch Time`, `Meters_train/val_*/...`, `Optim/lr_*`.

### 15.3 Resume

If `checkpoints/checkpoint.pt` exists, `Trainer.load_checkpoint()` resumes automatically (epoch, optimizer, scaler, `best_meter_values`). To resume from a specific checkpoint:

```yaml
trainer:
  checkpoint:
    resume_from: /abs/path/to/checkpoint_10.pt
    initialize_after_preemption: true
```

Or manually copy that file to `checkpoints/checkpoint.pt` before launch.

Checkpoint saving is **rank-0 only** and uses a `.tmp` + `mv` guard against corruption on preemption ([`sam3/train/trainer.py:387`](../sam3/sam3/train/trainer.py#L387)).

Set `skip_saving_ckpts: false` to actually save (RF100 template defaults to `true` for sweep mode — the PlantSeg config above fixes it).

---

## 16. VRAM / Speed Tips

| Knob | Effect |
|---|---|
| `resolution: 1008` → `672` or `512` | biggest VRAM win; but SAM 3 was trained at 1008, so keep 1008 if you can |
| `train_batch_size: 1`, `gradient_accumulation_steps: 4` | same effective batch, lower peak VRAM |
| `use_act_checkpoint_*: true` (vision/text/encoder/decoder) | saves ~30% VRAM at ~10% speed cost |
| `amp: bfloat16` | already on; do not disable |
| `num_train_workers: 8` | hide data loading; reduce if CPU-bound |
| `compile: true` (build_sam3_* with `compile=True`) | ~2× speedup on H100/A100 with `torch.compile` + FA3 (SAM 3.1) |
| Multiplex `multiplex_count: 16` | only for video; irrelevant for PlantSeg image task |

Expected per-GPU VRAM at 1008 + batch 1 + masks: ~18–22 GB (bf16, act ckpt on). Box-only: ~14–16 GB. Reduce resolution first if OOM.

---

## 17. Common Pitfalls

1. **`categories: []` empty** — trainer builds queries from categories; empty → 0 queries → `FilterEmptyTargets` drops every image → 0 batches. Fix with §10.1.
2. **`bpe_path` wrong** — tokenizer fails to load. Use `sam3/assets/bpe_simple_vocab_16e6.txt.gz` (installed with the package via `pkg_resources` fallback, but training config needs an absolute path).
3. **`enable_segmentation` mismatch** — `true` in model but `false` in collator (or vice versa) → shape mismatch. Keep all three in sync: `scratch.enable_segmentation`, `model.enable_segmentation`, `collate_fn.with_seg_masks`.
4. **`skip_saving_ckpts: true`** — you think training ran but `checkpoints/` is empty. Set to `false` for real runs (the PlantSeg config does).
5. **Absolute paths in config** — Hydra does not resolve `~`. Use absolute paths starting with `/`.
6. **COCO `bbox` format** — must be `xywh` (not `xyxy`). PlantSeg is already `xywh` — correct.
7. **115 classes but prompts are generic** — if every category name is `"lesion"`, the model cannot learn per-disease discrimination. Give each `category_id` a distinct phrase (`"apple black rot"`, `"tomato blight"`, …) or collapse to 1 class intentionally.
8. **`num_images: 100` leftover** — RF100 template limits to 100 images. Set `null` for full PlantSeg (5 367).

---

## 18. Reference: Key Files & Configs to Copy

| What | Path |
|---|---|
| Model builder | [`sam3/model_builder.py`](../sam3/sam3/model_builder.py) |
| Image processor | [`sam3/model/sam3_image_processor.py`](../sam3/sam3/model/sam3_image_processor.py) |
| Trainer | [`sam3/train/trainer.py`](../sam3/sam3/train/trainer.py) |
| Training entry | [`sam3/train/train.py`](../sam3/sam3/train/train.py) |
| Best config template | [`sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml`](../sam3/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml) |
| 10-shot template | [`sam3/train/configs/odinw13/odinw_text_only_train.yaml`](../sam3/sam3/train/configs/odinw13/odinw_text_only_train.yaml) |
| Dataset class | [`sam3/train/data/sam3_image_dataset.py`](../sam3/sam3/train/data/sam3_image_dataset.py) |
| COCO loader | [`sam3/train/data/coco_json_loaders.py`](../sam3/sam3/train/data/coco_json_loaders.py) |
| Loss wrapper | [`sam3/train/loss/sam3_loss.py`](../sam3/sam3/train/loss/sam3_loss.py) |
| Optim & schedulers | [`sam3/train/optim/optimizer.py`](../sam3/sam3/train/optim/optimizer.py), [`schedulers.py`](../sam3/sam3/train/optim/schedulers.py) |
| Ckpt utils | [`sam3/train/utils/checkpoint_utils.py`](../sam3/sam3/train/utils/checkpoint_utils.py) |
| Example notebooks | [`examples/sam3_image_predictor_example.ipynb`](../sam3/examples/sam3_image_predictor_example.ipynb), [`sam3.1_video_predictor_example.ipynb`](../sam3/examples/sam3.1_video_predictor_example.ipynb) |
| Training docs (upstream) | [`README_TRAIN.md`](../sam3/README_TRAIN.md) |
| Release notes | [`RELEASE_SAM3p1.md`](../sam3/RELEASE_SAM3p1.md) |

---

### Minimal end-to-end (cheat sheet)

```bash
# 1. fix categories
python scripts/prepare_plantseg_coco.py

# 2. install
cd sam3 && pip install -e ".[train]" && hf auth login

# 3. warm-start checkpoint path
python -c "from sam3.model_builder import download_ckpt_from_hf; print(download_ckpt_from_hf('sam3'))"

# 4. create sam3/train/configs/plantseg/plantseg_full_ft.yaml  (copy from §11, fix paths)

# 5. dry-run / sanity check (1 epoch, 10 images)
# edit plantseg_train.num_images: 10 and scratch.max_data_epochs: 1, then:
python sam3/train/train.py -c configs/plantseg/plantseg_full_ft.yaml --use-cluster 0 --num-gpus 1

# 6. full training
python sam3/train/train.py -c configs/plantseg/plantseg_full_ft.yaml --use-cluster 0 --num-gpus 1

# 7. monitor
tensorboard --logdir sam3_logs/plantseg/tensorboard

# 8. infer with fine-tuned ckpt
python - << 'PY'
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
model = build_sam3_image_model(checkpoint_path="sam3_logs/plantseg/checkpoints/checkpoint.pt", load_from_HF=False, enable_segmentation=True)
proc = Sam3Processor(model)
state = proc.set_image(Image.open("data/plantseg-v7/images/val/apple_black_rot_28.jpg"))
out = proc.set_text_prompt(state=state, prompt="black rot lesion")
print(out["boxes"][:3], out["scores"][:3])
PY
```

---

*Generated 2026-09-30 from repository + dataset inspection. Cross-check absolute paths before launching.*
