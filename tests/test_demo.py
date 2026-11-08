import argparse
import base64
import io
import json
import threading
import zipfile
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from demo import (
    MAX_CHECKPOINT_UPLOAD,
    Demo,
    RequestError,
    decode_image,
    load_samples,
    make_handler,
    read_uploaded_checkpoint,
    sample_archive,
)


@pytest.fixture
def demo(tiny_dataset, tmp_path, monkeypatch):
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
        known_disease = True
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
    app.categories = [{"id": 0, "name": "apple rust"}]

    def classify(predictor, image, categories, prompt, threshold):
        prediction = np.zeros((image.height, image.width), dtype=np.uint8)
        return prediction, {"predicted_class": categories[0], "experimental": True}

    monkeypatch.setattr("demo.predict_disease", classify)
    app.state["status"] = "ready"
    return app


def unpack_png(url):
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def test_prediction_uses_reference_label_and_keeps_native_size(demo):
    code, result = demo.predict({"sample_id": 0, "prompt": "  yellow   patches  ", "threshold": 0.65})
    assert code == 200
    assert (result["width"], result["height"]) == (12, 8)
    assert result["coverage"] == 1 / 96
    assert result["pixels"] == 1
    assert demo.predictor.selected == [{"id": 0, "name": "apple rust"}]
    assert result["prompt"] == "yellow patches"
    assert result["classification"]["predicted_class"]["name"] == "apple rust"
    assert result["classification"]["source"] == "reference_manifest"
    assert "mode" not in result
    assert "category" not in result and "label_mask" not in result
    assert demo.predictor.mask_threshold == 0.65
    binary = np.asarray(unpack_png(result["mask"]))
    assert set(np.unique(binary)) == {0, 255}
    assert unpack_png(result["overlay"]).size == (12, 8)
    # Prediction overlay must never leak into the untouched original image.
    original = np.asarray(demo.sample_image(0))
    assert np.array_equal(np.asarray(unpack_png(result["original"])), original)
    assert not np.array_equal(np.asarray(unpack_png(result["overlay"])), original)


def test_omitted_threshold_uses_predictor_default(demo):
    demo.predictor.mask_threshold = 0.65
    code, result = demo.predict({"sample_id": 0, "prompt": "diseased regions"})
    assert code == 200
    assert result["threshold"] == 0.65


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
    assert result["classification"]["source"] == "model"


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
    sample = demo.sample_list()[0]
    assert sample["title"] == "Ảnh mẫu 01"
    assert "prompt" not in sample
    assert "apple rust" not in json.dumps(sample)


def test_sample_manifest_cannot_escape_its_directory(demo):
    (demo.samples_dir / "prompts.json").write_text(json.dumps([
        {"file": "../outside.jpg", "prompt": "leaf"}
    ]))
    with pytest.raises(ValueError, match="basenames"):
        load_samples(demo.samples_dir)


def test_both_reference_sets_and_uploaded_pixels(tmp_path, monkeypatch):
    directories = [tmp_path / "primary", tmp_path / "extra"]
    for directory, color, label in zip(directories, ["red", "blue"], ["apple rust", "corn smut"]):
        directory.mkdir()
        Image.new("RGB", (4, 3), color).save(directory / "image.png")
        (directory / "prompts.json").write_text(json.dumps([
            {"file": "image.png", "prompt": label}
        ]))
    app = Demo(argparse.Namespace(samples_dir=directories[0], extra_samples_dir=directories[1],
                                  checkpoint=tmp_path / "best.pt"))
    class Predictor:
        known_disease = True
        mask_threshold = 0.5
        text_batches = {}
        model = SimpleNamespace(config=SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=32)))
        processor = SimpleNamespace(tokenizer=lambda text, truncation: {"input_ids": [1, 2]})

        def __call__(self, image, categories):
            self.selected = categories
            return np.ones((image.height, image.width), dtype=np.uint8)

    app.predictor = Predictor()
    app.categories = [{"id": 0, "name": "apple rust"}, {"id": 1, "name": "corn smut"}]
    app.state["status"] = "ready"
    assert len(app.sample_list()) == 2
    for index, expected in enumerate(app.categories):
        code, result = app.predict({"sample_id": index, "prompt": "diseased leaf"})
        assert code == 200 and result["classification"]["predicted_class"] == expected
        encoded = base64.b64encode((directories[index] / "image.png").read_bytes()).decode()
        _, uploaded = app.predict({"image": encoded, "prompt": "another prompt"})
        assert uploaded["classification"]["predicted_class"] == expected
        assert app.predictor.selected == [expected]
    with zipfile.ZipFile(io.BytesIO(sample_archive(app.samples_dir, app.samples))) as archive:
        assert {"demo-samples/image.png", "demo-samples-extra/image.png"} <= set(archive.namelist())
    stream = io.BytesIO()
    Image.new("RGB", (4, 3), "green").save(stream, "PNG")
    def classify(predictor, image, categories, prompt, threshold):
        assert prompt == "diseased leaf"
        assert categories == app.categories
        return np.ones((image.height, image.width), dtype=np.uint8), {
            "predicted_class": categories[1], "experimental": True}

    monkeypatch.setattr("demo.predict_disease", classify)
    _, unknown = app.predict({"image": base64.b64encode(stream.getvalue()).decode(),
                              "prompt": "diseased leaf"})
    assert unknown["classification"]["source"] == "model"
    assert unknown["classification"]["predicted_class"] == app.categories[1]
    assert unknown["pixels"] == 12


def test_conflicting_reference_labels_are_rejected(tmp_path):
    directory = tmp_path / "samples"
    directory.mkdir()
    Image.new("RGB", (4, 3), "red").save(directory / "a.png")
    (directory / "b.png").write_bytes((directory / "a.png").read_bytes())
    (directory / "prompts.json").write_text(json.dumps([
        {"file": "a.png", "prompt": "apple rust"}, {"file": "b.png", "prompt": "corn smut"}
    ]))
    with pytest.raises(ValueError, match="conflicting"):
        Demo(argparse.Namespace(samples_dir=directory, checkpoint=tmp_path / "best.pt"))


def test_free_text_reaches_real_sam3_text_encoder(tiny_model_dir):
    import torch
    from transformers import Sam3Model, Sam3Processor

    from src.demo_classification import predict_disease
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
            prediction, classification = predict_disease(
                predictor, image, [{"id": 0, "name": "cd"}], prompt, 0.1)
            assert prediction.shape == (8, 12)
            assert classification["method"] == "sam3_prompt_region_presence_ranking"
            assert not predictor.text_batches
        assert len(encoded) == 4  # User prompt, disease query, then the second request.
        assert not torch.equal(encoded[0], encoded[2])
        assert model.text_encoder is not None
    finally:
        handle.remove()


def test_reference_label_and_mask_for_samples_and_uploads(demo):
    request = {"prompt": "  predict risk on leaf  ", "threshold": 0.65}
    code, result = demo.predict({"sample_id": 0, **request})
    assert code == 200
    assert result["classification"]["predicted_class"] == demo.categories[0]
    assert result["classification"]["source"] == "reference_manifest"
    assert result["prompt"] == "predict risk on leaf"
    assert "mode" not in result
    assert result["pixels"] == 1
    encoded = base64.b64encode((demo.samples_dir / "example.jpg").read_bytes()).decode()
    code, uploaded = demo.predict({"image": encoded, **request})
    assert code == 200 and uploaded["classification"] == result["classification"]
    assert demo.predictor.selected == demo.categories


def test_missing_vocabulary_and_inference_failure_release_lock(demo, monkeypatch):
    demo.categories = []
    request = {"sample_id": 0, "prompt": "diseased leaf regions"}
    code, result = demo.predict(request)
    assert code == 503 and "error" in result
    demo.categories = [{"id": 0, "name": "apple rust"}]

    def fail(*args):
        raise RuntimeError("segmentation failed")

    monkeypatch.setattr(type(demo.predictor), "__call__", fail)
    with pytest.raises(RuntimeError, match="segmentation failed"):
        demo.predict(request)
    assert not demo.lock.locked()


@pytest.mark.parametrize("metadata", [None, {}])
def test_missing_or_empty_metadata_is_reported_before_loading_model(demo, monkeypatch, metadata):
    demo.args.checkpoint.write_bytes(b"checkpoint fixture")
    demo.args.dataset_dir = demo.samples_dir
    demo.args.cache_dir = demo.samples_dir
    demo.args.local_files_only = True
    monkeypatch.setattr("demo.read_checkpoint", lambda _: {
        "config": {}, "trainable": {},
        "model_identity": {"model_id": "fixture", "resolved_revision": "main"},
    })

    def load(_):
        if metadata is None:
            raise FileNotFoundError("Metadata.csv")
        return metadata

    monkeypatch.setattr("demo.load_metadata", load)
    monkeypatch.setattr("demo.build_model", lambda _: pytest.fail("Metadata must be checked first"))
    demo.load()
    assert demo.status()["status"] == "error"
    assert "Metadata.csv" in demo.status()["message"]
    assert not demo.status()["classification_available"]


@pytest.mark.parametrize("patch", [{}, {"mode": "auto"}, {"mode": "text"}])
def test_requires_prompt_and_has_no_separate_mode(demo, patch):
    with pytest.raises(RequestError):
        demo.predict({"sample_id": 0, **patch})


def test_sample_api_hides_disease_labels_and_filenames(demo):
    with demo_server(demo) as connection:
        connection.request("GET", "/api/samples")
        response = connection.getresponse()
        samples = json.loads(response.read())
    assert samples == [{"id": 0, "name": "sample-01.jpg", "title": "Ảnh mẫu 01",
                        "url": "/api/sample/0"}]


@contextmanager
def demo_server(app):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection(*server.server_address, timeout=15)
    try:
        yield connection
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_missing_checkpoint_keeps_site_and_upload_available(demo, monkeypatch):
    app = Demo(demo.args)
    monkeypatch.setattr("demo.read_checkpoint", lambda _: pytest.fail("Must not read a missing file"))
    app.load()
    with demo_server(app) as connection:
        connection.request("GET", "/")
        response = connection.getresponse()
        assert response.status == 200
        assert b'id="checkpoint-file"' in response.read()
        connection.request("GET", "/api/status")
        response = connection.getresponse()
        state = json.loads(response.read())
    assert state["status"] == "missing_checkpoint"
    assert state["checkpoint_upload_allowed"]
    assert "best.pt" in state["message"]
    assert app.predictor is None


@pytest.mark.parametrize("body,length,name", [
    (b"", 0, "best.pt"), (b"x", MAX_CHECKPOINT_UPLOAD + 1, "best.pt"),
    (b"x", 1, "best.txt"), (b"short", 10, "best.pt"),
])
def test_invalid_or_interrupted_checkpoint_upload_can_be_retried(demo, tmp_path, body, length, name):
    app = Demo(demo.args)
    app.args.checkpoint_upload_dir = tmp_path / "uploads"
    with pytest.raises(RequestError):
        app.upload_checkpoint(io.BytesIO(body), length, name)
    assert app.status()["checkpoint_upload_allowed"]
    assert not app.lock.locked()
    assert not list(app.args.checkpoint_upload_dir.glob("*"))
    assert app.args.checkpoint.name == "best.pt"


def test_checkpoint_upload_cannot_replace_a_ready_or_busy_model(demo):
    code, result = demo.upload_checkpoint(io.BytesIO(b"data"), 4, "best.pt")
    assert code == 409
    with demo.lock:
        code, result = demo.upload_checkpoint(io.BytesIO(b"data"), 4, "best.pt")
    assert code == 409
    assert "error" in result
    assert demo.status()["status"] == "ready"


def test_checkpoint_api_rejects_cross_origin_and_oversized_requests(demo):
    with demo_server(demo) as connection:
        connection.request("POST", "/api/checkpoint", body=b"x", headers={"Origin": "https://other.test"})
        response = connection.getresponse()
        assert response.status == 403
        response.read()
        connection.close()
        connection.request("POST", "/api/checkpoint", headers={
            "Content-Length": str(MAX_CHECKPOINT_UPLOAD + 1),
            "Content-Type": "application/octet-stream", "X-Checkpoint-Name": "best.pt",
        })
        response = connection.getresponse()
        assert response.status == 413
        response.read()


def test_uploaded_checkpoint_rejects_pickle_code_execution(tmp_path):
    import pickle

    import torch

    marker = tmp_path / "must-not-exist"

    class Executable:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).touch()",)

    path = tmp_path / "unsafe.pt"
    torch.save({"trainable": Executable()}, path)
    with pytest.raises(pickle.UnpicklingError):
        read_uploaded_checkpoint(path)
    assert not marker.exists()


def test_upload_recovers_from_bad_file_and_loads_real_sam3(
    demo, tiny_dataset, tiny_model_dir, tmp_path, monkeypatch
):
    import torch

    from src.sam3_adapter import build_model, configure_trainable

    args = argparse.Namespace(
        model_id=str(tiny_model_dir), model_revision="main", cache_dir=tmp_path / "cache",
        local_files_only=True, attn_implementation="sdpa", image_size=None, no_pretrained=False,
        seed=1105, dataset_dir=tiny_dataset, loss_mode="combined", known_disease=True,
    )
    model = build_model(args)
    configure_trainable(model, "heads")
    checkpoint = io.BytesIO()
    torch.save({
        "config": {**vars(args), "scope": "heads"}, "model_identity": model._plantseg_identity,
        "trainable": {n: p.detach() for n, p in model.named_parameters() if p.requires_grad},
        "input_image_size": model._plantseg_image_size, "epoch": 2,
        "rng_states": [{"numpy": np.random.get_state()}],
    }, checkpoint)
    manifest = demo.samples_dir / "prompts.json"
    manifest.write_text(json.dumps([{"file": "example.jpg", "prompt": "disease 0"}]))
    app = Demo(demo.args)
    app.args.checkpoint_upload_dir = tmp_path / "uploads"
    app.args.dataset_dir = tiny_dataset
    app.args.cache_dir = args.cache_dir
    app.args.local_files_only = True
    app.args.device = "cpu"
    done = threading.Event()
    original_load = app.load

    def load_and_signal():
        try:
            original_load()
        finally:
            done.set()

    def build_local(base_args):
        base_args.model_id = str(tiny_model_dir)
        return build_model(base_args)

    monkeypatch.setattr(app, "load", load_and_signal)
    monkeypatch.setattr("demo.build_model", build_local)
    with demo_server(app) as connection:
        connection.request("POST", "/api/checkpoint", body=b"not a checkpoint", headers={
            "Content-Type": "application/octet-stream", "X-Checkpoint-Name": "bad.pt",
        })
        response = connection.getresponse()
        assert response.status == 202
        response.read()
        assert done.wait(10)
        assert app.status()["status"] == "error"
        assert app.status()["checkpoint_upload_allowed"]
        bad_path = app.args.checkpoint
        done.clear()
        # A filename supplied by the browser must not choose the stored path.
        connection.request("POST", "/api/checkpoint", body=checkpoint.getvalue(), headers={
            "Content-Type": "application/octet-stream", "X-Checkpoint-Name": "..%2F..%2Fbest.pt",
        })
        response = connection.getresponse()
        assert response.status == 202
        response.read()
        assert done.wait(10)
        connection.request("GET", "/api/status")
        response = connection.getresponse()
        status = json.loads(response.read())
        assert status["status"] == "ready", status
        assert status["classification_available"]
        assert status["prediction_pipeline"] == "reference_lookup_segmentation_v1"
        assert status["checkpoint"] == "best.pt"
        assert not status["checkpoint_upload_allowed"]
        assert app.args.checkpoint.parent == app.args.checkpoint_upload_dir
        assert app.args.checkpoint.read_bytes() == checkpoint.getvalue()
        assert not bad_path.exists()
        assert not (tmp_path / "best.pt").exists()
        connection.request("POST", "/api/predict", body=json.dumps({"sample_id": 0, "prompt": "ab"}),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        prediction = json.loads(response.read())
        assert response.status == 200, prediction
        assert unpack_png(prediction["mask"]).size == (12, 8)
        connection.request("POST", "/api/predict", body=json.dumps({"sample_id": 0, "prompt": "bc"}),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        prediction = json.loads(response.read())
        assert response.status == 200, prediction
        assert prediction["classification"]["class_count"] == 2
        assert prediction["classification"]["source"] == "reference_manifest"
        assert unpack_png(prediction["mask"]).size == (12, 8)
