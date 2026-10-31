import csv
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def tiny_dataset(tmp_path):
    root = tmp_path / "plantseg"
    rows = []
    for split in ("train", "val", "test"):
        for i in range(3):
            name = f"{split}_{i}.jpg"
            image_dir, mask_dir = root / "images" / split, root / "annotations" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (12, 8), (20, 60, 30)).save(image_dir / name)
            mask = np.zeros((8, 12), np.uint8)
            mask[1:3, 1:4] = i % 2 + 1
            mask[5:7, 7:10] = i % 2 + 1
            Image.fromarray(mask).save(mask_dir / Path(name).with_suffix(".png"))
            rows.append(
                {
                    "Name": name,
                    "Index": i % 2,
                    "Plant": "apple",
                    "Disease": f"disease {i % 2}",
                    "Label file": Path(name).with_suffix(".png").name,
                }
            )
    with (root / "Metadata.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


@pytest.fixture
def tiny_model_dir(tmp_path):
    """A real Transformers SAM3 + processor saved locally as safetensors; no HF access."""
    import json
    from transformers import (
        Sam3Config,
        Sam3Model,
        Sam3ImageProcessor,
        Sam3Processor,
        CLIPTextConfig,
        CLIPTokenizer,
    )
    from transformers.models.sam3.configuration_sam3 import (
        Sam3VisionConfig,
        Sam3ViTConfig,
        Sam3GeometryEncoderConfig,
        Sam3DETREncoderConfig,
        Sam3DETRDecoderConfig,
        Sam3MaskDecoderConfig,
    )
    import torch

    torch.manual_seed(71)
    config = Sam3Config(
        vision_config=Sam3VisionConfig(
            backbone_config=Sam3ViTConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                image_size=32,
                patch_size=8,
                pretrain_image_size=32,
                window_size=2,
                global_attn_indexes=[0],
            ),
            fpn_hidden_size=32,
            backbone_feature_sizes=[[16, 16], [8, 8], [4, 4]],
        ),
        text_config=CLIPTextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=8,
            projection_dim=16,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=0,
        ),
        geometry_encoder_config=Sam3GeometryEncoderConfig(
            hidden_size=32, num_layers=1, num_attention_heads=4, intermediate_size=64
        ),
        detr_encoder_config=Sam3DETREncoderConfig(
            hidden_size=32, num_layers=1, num_attention_heads=4, intermediate_size=64, dropout=0
        ),
        detr_decoder_config=Sam3DETRDecoderConfig(
            hidden_size=32,
            num_layers=1,
            num_queries=4,
            num_attention_heads=4,
            intermediate_size=64,
            dropout=0,
        ),
        mask_decoder_config=Sam3MaskDecoderConfig(hidden_size=32, num_attention_heads=4),
    )
    root = tmp_path / "sam3-tiny"
    root.mkdir()
    Sam3Model(config).save_pretrained(root, safe_serialization=True)
    vocabulary = {"<|startoftext|>": 0, "<|endoftext|>": 1}
    vocabulary.update({letter: i + 2 for i, letter in enumerate("abcdefghijklmnopqrstuvwxyz")})
    (root / "vocab.json").write_text(json.dumps(vocabulary))
    (root / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(
        vocab=vocabulary,
        merges=[],
        model_max_length=8,
        pad_token="<|startoftext|>",
    )
    processor = Sam3Processor(
        image_processor=Sam3ImageProcessor(
            size={"height": 32, "width": 32}, mask_size={"height": 16, "width": 16}
        ),
        tokenizer=tokenizer,
    )
    processor.save_pretrained(root)
    return root
