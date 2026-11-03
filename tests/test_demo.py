import argparse
import base64
import io
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from demo import Demo, RequestError, decode_image, load_samples, predict_text, sample_archive


@pytest.fixture
def demo(tiny_dataset, tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "example.jpg").write_bytes((tiny_dataset / "images/test/test_0.jpg").read_bytes())
    (samples / "example.txt").write_text("apple rust\n")
    (samples / "prompts.json").write_text(json.dumps([
        {"file": "example.jpg", "prompt": "apple rust", "title": "Ảnh mẫu"}
    ]))
    app = Demo(argparse.Namespace(samples_dir=samples, checkpoint=tmp_path / "best.pt"))

    class Predictor:
        mask_threshold = 0.5
        model = SimpleNamespace(config=SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=32)))
        processor = SimpleNamespace(tokenizer=lambda text, truncation: {
            "input_ids": list(range(len(text.split()) + 2))})
        text_batches = {}

        def __call__(self, image, categories):
            self.selected = categories
            prediction = np.zeros((image.height, image.width), dtype=np.uint8)
            prediction[0, 0] = categories[0]["id"] + 1
            return prediction

    app.predictor = Predictor()
    app.state["status"] = "ready"
    return app


def unpack_png(url):
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def test_prediction_uses_free_text_and_keeps_native_size(demo):
    code, result = demo.predict({"sample_id": 0, "prompt": "  yellow   patches  ", "threshold": 0.65})
    assert code == 200
    assert (result["width"], result["height"]) == (12, 8)
    assert result["coverage"] == 1 / 96
    assert result["pixels"] == 1
    assert demo.predictor.selected == [{"id": 0, "name": "yellow patches"}]
    assert result["prompt"] == "yellow patches"
    assert "category" not in result and "label_mask" not in result
    assert demo.predictor.mask_threshold == 0.65
    binary = np.asarray(unpack_png(result["mask"]))
    assert set(np.unique(binary)) == {0, 255}
    assert unpack_png(result["overlay"]).size == (12, 8)
    # Prediction overlay must never leak into the untouched original image.
    original = np.asarray(demo.sample_image(0))
    assert np.array_equal(np.asarray(unpack_png(result["original"])), original)
    assert not np.array_equal(np.asarray(unpack_png(result["overlay"])), original)


@pytest.mark.parametrize("patch", [
    {"prompt": None}, {"prompt": True}, {"prompt": "  "},
    {"prompt": "a" * 201}, {"prompt": "a\0b"}, {"prompt": "a " * 40},
    {"threshold": float("nan")}, {"threshold": 1}, {"threshold": "0.5"},
    {"sample_id": -1}, {"sample_id": True},
])
def test_rejects_invalid_inference_inputs(demo, patch):
    with pytest.raises(RequestError):
        demo.predict({"sample_id": 0, "prompt": "apple rust", **patch})
    assert not demo.lock.locked()


def test_busy_model_does_not_run_concurrent_inference(demo):
    with demo.lock:
        code, result = demo.predict({"sample_id": 0, "prompt": "apple rust"})
    assert code == 409
    assert "error" in result
    assert demo.lock.acquire(blocking=False)
    demo.lock.release()


def test_loading_model_returns_retryable_error(demo):
    demo.state["status"] = "loading"
    code, result = demo.predict({"sample_id": 0, "prompt": "apple rust"})
    assert code == 503
    assert "error" in result


def test_uploaded_image_and_exif_are_normalized(demo):
    image = Image.new("RGB", (12, 8), "green")
    exif = image.getexif()
    exif[274] = 6
    stream = io.BytesIO()
    image.save(stream, "JPEG", exif=exif)
    encoded = base64.b64encode(stream.getvalue()).decode()
    code, result = demo.predict({"image": encoded, "prompt": "apple rust"})
    assert code == 200
    assert (result["width"], result["height"]) == (8, 12)


def test_corrupt_upload_is_rejected_without_locking_model(demo):
    for encoded in ("invalid base64!", base64.b64encode(b"not an image").decode()):
        with pytest.raises(RequestError):
            demo.predict({"image": encoded, "prompt": "apple rust"})
    assert not demo.lock.locked()
    with pytest.raises(RequestError):
        decode_image(b"x" * (12 * 1024 * 1024 + 1))


def test_inference_failure_releases_lock_and_prompt_cache(demo, monkeypatch):
    def fail(*args):
        demo.predictor.text_batches["prompt"] = "device tensors"
        raise RuntimeError("inference failed")

    monkeypatch.setattr(type(demo.predictor), "__call__", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        demo.predict({"sample_id": 0, "prompt": "apple rust"})
    assert not demo.lock.locked()
    assert not demo.predictor.text_batches


def test_archive_pairs_the_original_image_with_its_prompt(demo):
    with zipfile.ZipFile(io.BytesIO(sample_archive(demo.samples_dir, demo.samples))) as archive:
        assert archive.read("demo-samples/example.jpg") == (
            demo.samples_dir / "example.jpg").read_bytes()
        assert archive.read("demo-samples/example.txt").decode().strip() == "apple rust"
        entries = json.loads(archive.read("demo-samples/prompts.json"))
        assert entries[0]["file"] == "example.jpg"
    assert demo.sample_list()[0]["prompt"] == "apple rust"


def test_sample_manifest_cannot_escape_its_directory(demo):
    (demo.samples_dir / "prompts.json").write_text(json.dumps([
        {"file": "../outside.jpg", "prompt": "leaf"}
    ]))
    with pytest.raises(ValueError, match="basenames"):
        load_samples(demo.samples_dir)


def test_free_text_reaches_real_sam3_text_encoder(tiny_model_dir):
    import torch
    from transformers import Sam3Model, Sam3Processor

    from src.evaluation import Sam3SemanticPredictor

    model = Sam3Model.from_pretrained(tiny_model_dir, local_files_only=True).eval()
    model._plantseg_processor = Sam3Processor.from_pretrained(tiny_model_dir, local_files_only=True)
    predictor = Sam3SemanticPredictor(
        model, torch.device("cpu"), None, 0.5, 0.5, mask_source="semantic", known_disease=True)
    encoded = []

    def record(module, args, kwargs):
        encoded.append(kwargs["input_ids"].clone())

    handle = model.text_encoder.register_forward_pre_hook(record, with_kwargs=True)
    try:
        image = Image.new("RGB", (12, 8), "green")
        # Tiny fixture vocabulary contains plain letters; two-letter words retain
        # a distinct prefix token (single-letter end-of-word tokens are unknown).
        for prompt in ("ab", "bc"):
            prediction = predict_text(predictor, image, prompt, 0.5)
            assert prediction.shape == (8, 12)
            assert not predictor.text_batches
        assert len(encoded) == 2
        assert not torch.equal(encoded[0], encoded[1])
        assert model.text_encoder is not None
    finally:
        handle.remove()
