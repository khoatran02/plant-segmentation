# Evidence map

Inspected 2026-10-30. Paths below are relative to the project root unless prefixed with `reports/`.

Archive note (2026-11-01): the table maps the historical manuscript to the implementation inspected on October 30. The upstream checkout, Hydra guide and obsolete `collect_evidence.py` have since been removed. Current source layout and usage are documented in the project README.

| Manuscript topic | Evidence |
|---|---|
| Local v6/v7 file and metadata totals | `reports/collect_evidence.py`, `reports/evidence/snapshot.json` |
| Orphan records, effective polygon image/annotation counts | Actual `build_coco_split` functions extracted from `src/training.py` and executed on local files in temporary directories |
| Pixel audit, duplicate clusters, polygon mismatches, EXIF findings | `docs/plantseg-v6-v7-statistics.md`, dated 2026-10-07; archived copy under `reports/evidence/`; these analyses were not rerun |
| Model example image montage | `docs/plantseg-stats/samples.jpg`, copied unchanged to `reports/figures/dataset_samples.jpg`; ground truth only |
| Category names and IDs | `src/training.py`: `load_metadata`, `build_coco_split` |
| Prompt sampling | `src/training.py`: `PlantSegLoader` |
| Transforms and batching | `src/training.py`: `build_transforms`, `build_dataset`, `make_loader`; vendored `sam3/train/transforms/basic.py` confirms square resizing |
| Trainable scope and text caching | `src/training.py`: `configure_trainable`, `cache_text_features` |
| Loss weights and matchers | `src/training.py`: `build_criterion`; vendored `sam3/train/loss/loss_fns.py` |
| Optimizer, precision, checkpoints and defaults | `src/training.py`: `build_optimizer`, `train`, `save_checkpoint`, `parse_args` |
| Current validation metrics and threshold rules | `src/training.py`: `evaluate` |
| Additional native-frame PNG metrics, comparisons and cleaning protocol | Experimental proposals in this paper; not implemented or measured by this report |

Primary literature verified online:

- PlantSeg original preprint: https://arxiv.org/abs/2409.04038 (Wei et al., 2024).
- v7 dataset record: https://zenodo.org/records/17719108 (published November 25, 2025).
- SAM 3: https://arxiv.org/abs/2511.16719 (Carion et al., 2025; revised March 28, 2026). The project builder's image pathway downloads `version="sam3"`; SAM 3.1 video tracking is not the method documented here.
- U-Net: https://arxiv.org/abs/1505.04597 (Ronneberger et al., 2015).
- DeepLabv3+: https://arxiv.org/abs/1802.02611 (Chen et al., 2018).

`docs/sam3-guide.md` is an earlier planning document. Where its Hydra recipe differs from `src/training.py`, the manuscript describes the actual Python script. No hardware speed or memory estimates from that guide are treated as measurements.
