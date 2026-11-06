"""Local HTML demo for a trained PlantSeg SAM3 adaptation (no web dependencies)."""

from __future__ import annotations

import argparse
import base64
import binascii
import gc
import hashlib
import io
import json
import logging
import math
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from numpy._core.multiarray import _reconstruct
from PIL import Image, ImageOps, UnidentifiedImageError

from src.data import DEFAULT_DATASET_DIR, categories_from_metadata, load_metadata
from src.demo_classification import predict_disease
from src.evaluation import Sam3SemanticPredictor
from src.runtime import read_checkpoint
from src.sam3_adapter import (
    build_model,
    configure_semantic_only,
    configure_trainable,
)

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 12 * 1024 * 1024
MAX_CHECKPOINT_UPLOAD = 4 * 1024 * 1024 * 1024
CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".ckpt"}
MAX_PIXELS = 20_000_000
MAX_PROMPT_LENGTH = 200
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
LOG = logging.getLogger("plantseg.demo")


class RequestError(ValueError):
    pass


def read_uploaded_checkpoint(path):
    """Read project checkpoints without executing arbitrary uploaded pickle globals."""
    allowed = [np.ndarray, np.dtype, type(np.dtype(np.uint32)), _reconstruct,
               (_reconstruct, "numpy.core.multiarray._reconstruct")]
    # Checkpoints saved with Python 3.13 use pathlib._local; Kaggle uses PosixPath.
    for name in ("PosixPath", "WindowsPath", "PurePosixPath", "PureWindowsPath"):
        cls = pathlib.PurePath if name.startswith("Pure") else pathlib.Path
        allowed.extend((cls, f"{module}.{name}") for module in ("pathlib", "pathlib._local"))
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def validate_prompt(value):
    if not isinstance(value, str) or not value.strip():
        raise RequestError("Nhập text prompt mô tả vùng bạn muốn phân đoạn.")
    if len(value) > MAX_PROMPT_LENGTH:
        raise RequestError(f"Prompt quá dài. Hãy dùng tối đa {MAX_PROMPT_LENGTH} ký tự.")
    if any(ord(char) < 32 and char not in "\t\r\n" for char in value):
        raise RequestError("Prompt chứa ký tự điều khiển không hợp lệ.")
    return " ".join(value.split())


def load_samples(directory):
    """Load only the explicit, portable examples, independent of the dataset split."""
    entries = json.loads((directory / "prompts.json").read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Sample manifest must contain at least one image/prompt pair")
    for entry in entries:
        name = entry["file"]
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Sample filenames must be basenames")
        path = directory / name
        if not path.is_file() or path.resolve().parent != directory.resolve():
            raise ValueError(f"Sample image not found in sample directory: {name}")
        entry["prompt"] = validate_prompt(entry["prompt"])
        entry["image_path"] = path
    return entries


def sample_archive(directory, samples):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        directories = [directory, *dict.fromkeys(
            s["image_path"].parent for s in samples if s["image_path"].parent != directory
        )]
        for index, sample_dir in enumerate(directories):
            group = [s for s in samples if s["image_path"].parent == sample_dir]
            names = {s["file"] for s in group} | {"README.md", "prompts.json"}
            names |= {str(Path(s["file"]).with_suffix(".txt")) for s in group}
            folder = "demo-samples" if index == 0 else f"demo-samples-extra{index if index > 1 else ''}"
            for name in sorted(names):
                path = sample_dir / name
                if path.is_file() and path.resolve().is_relative_to(sample_dir.resolve()):
                    archive.write(path, f"{folder}/{name}")
    return stream.getvalue()


def validate_prompt_tokens(predictor, prompt):
    if getattr(predictor.model, "_plantseg_text_hidden", None) is not None:
        raise RuntimeError("Free text requires a live text encoder, not a fixed vocabulary cache")
    maximum = predictor.model.config.text_config.max_position_embeddings
    tokens = predictor.processor.tokenizer(prompt, truncation=False)["input_ids"]
    if len(tokens) > maximum:
        raise RequestError(
            f"Prompt vượt giới hạn {maximum} token của text encoder. Hãy dùng cụm từ ngắn hơn."
        )


def predict_text(predictor, image, prompt, threshold):
    """Low-level prompt mask helper; the demo API also predicts the disease class."""
    validate_prompt_tokens(predictor, prompt)
    predictor.mask_threshold = threshold
    try:
        # Label 1 means the concept in this request, not a disease class prediction.
        return predictor(image, [{"id": 0, "name": prompt}])
    finally:
        # User-entered prompts are unbounded; release their device tensors after each request.
        predictor.text_batches.clear()


def png_url(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def mask_overlay(image, foreground, color):
    pixels = np.asarray(image).copy()
    pixels[foreground] = (pixels[foreground] * 0.35 + np.array(color) * 0.65).astype(np.uint8)
    return Image.fromarray(pixels)


def decode_image(raw):
    if len(raw) > MAX_UPLOAD:
        raise RequestError("Ảnh vượt quá 12 MB. Vui lòng chọn ảnh nhỏ hơn.")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise RequestError("Chỉ hỗ trợ ảnh JPG, PNG hoặc WebP.")
            if source.width * source.height > MAX_PIXELS:
                raise RequestError("Ảnh vượt quá 20 megapixel.")
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise RequestError("Không đọc được ảnh. Hãy dùng ảnh JPG, PNG hoặc WebP hợp lệ.") from error


def image_fingerprint(image):
    """Hash displayed RGB pixels and dimensions, independent of filename/metadata."""
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.digest()


def build_reference_index(samples):
    references = {}
    for sample in samples:
        fingerprint = image_fingerprint(decode_image(sample["image_path"].read_bytes()))
        label = sample["prompt"]
        if fingerprint in references and references[fingerprint] != label:
            raise ValueError("Identical reference images have conflicting disease labels")
        references[fingerprint] = label
    return references


class Demo:
    def __init__(self, args):
        self.args = args
        self.samples_dir = args.samples_dir
        self.samples = load_samples(self.samples_dir)
        extra_dir = getattr(args, "extra_samples_dir", None)
        if extra_dir and extra_dir.is_dir() and extra_dir.resolve() != self.samples_dir.resolve():
            self.samples.extend(load_samples(extra_dir))
        self.reference_index = build_reference_index(self.samples)
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.state = {"status": "loading", "message": "Đang nạp SAM 3 và checkpoint…",
                      "checkpoint": args.checkpoint.name}
        self.predictor = None
        self.categories = []
        self.uploaded_checkpoint = False
        if not args.checkpoint.is_file():
            self.missing_checkpoint()

    def missing_checkpoint(self):
        with self.state_lock:
            self.state.update(status="missing_checkpoint", message=(
                f"Không tìm thấy checkpoint {self.args.checkpoint.name}. "
                "Hãy tải checkpoint đã fine-tune lên để sử dụng demo."))

    def status(self):
        with self.state_lock:
            return {**self.state, "prompt_mode": "classify_and_segment", "max_prompt_length": MAX_PROMPT_LENGTH,
                    "prediction_pipeline": "reference_lookup_segmentation_v1",
                    "classification_available": bool(self.categories),
                    "checkpoint_upload_allowed": self.state["status"] in {"missing_checkpoint", "error"},
                    "max_checkpoint_bytes": MAX_CHECKPOINT_UPLOAD}

    def load(self):
        with self.lock:
            self._load()

    def _load(self):
        if not self.args.checkpoint.is_file():
            self.missing_checkpoint()
            return
        stage = "checkpoint"
        try:
            with self.state_lock:
                self.state.update(status="loading", message="Đang nạp SAM 3 và checkpoint…")
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            reader = read_uploaded_checkpoint if self.uploaded_checkpoint else read_checkpoint
            state = reader(self.args.checkpoint)
            config, identity = state["config"], state["model_identity"]
            adaptation = state["trainable"]
            size = state.get("input_image_size")
            epoch = state.get("epoch")
            del state
            gc.collect()
            known = config.get("known_disease", False)
            args = argparse.Namespace(
                model_id=identity["model_id"], model_revision=identity["resolved_revision"],
                cache_dir=self.args.cache_dir, local_files_only=self.args.local_files_only,
                attn_implementation=config.get("attn_implementation", "sdpa"),
                image_size=size[0] if size else None, no_pretrained=False,
                seed=config.get("seed", 1105), dataset_dir=self.args.dataset_dir,
                loss_mode=config.get("loss_mode", "combined"), known_disease=known,
            )
            stage = "metadata"
            categories = categories_from_metadata(load_metadata(self.args.dataset_dir))
            if not categories:
                raise ValueError("Metadata.csv contains no disease classes")
            stage = "base_model"
            model = build_model(args)
            stage = "checkpoint"
            saved, current = dict(identity), dict(model._plantseg_identity)
            # Legacy checkpoints record 'main'; mirror inference.py's compatibility policy.
            if saved["resolved_revision"] == "main":
                current["resolved_revision"] = "main"
            if current != saved:
                raise ValueError("Checkpoint/base identity mismatch (revision, vocabulary or backend).")
            configure_trainable(model, config["scope"])
            expected = {n for n, p in model.named_parameters() if p.requires_grad}
            if expected != set(adaptation):
                raise ValueError("Checkpoint parameter set differs from the pretrained model.")
            incompatible = model.load_state_dict(adaptation, strict=False)
            if incompatible.unexpected_keys:
                raise ValueError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
            del adaptation
            # Keep the CLIP text encoder, including for heads-only checkpoints. The
            # training/evaluation cache drops it and cannot encode a new user prompt.
            mask_source = "semantic" if known else config.get("eval_mask_source", "instance")
            if mask_source == "semantic":
                configure_semantic_only(model)
            stage = "device"
            device = self.args.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else (
                    "mps" if torch.backends.mps.is_available() else "cpu")
            device = torch.device(device)
            model = model.to(device).eval()
            model.requires_grad_(False)
            dtype = torch.float16 if device.type == "cuda" else None
            self.predictor = Sam3SemanticPredictor(
                model, device, dtype, config.get("score_threshold", 0.5),
                config.get("mask_threshold", 0.5), 1, mask_source, known,
            )
            self.categories = categories
            gc.collect()
            with self.state_lock:
                self.state.update(status="ready", message="Model sẵn sàng", device=str(device),
                                  known_disease=known, input_size=model._plantseg_image_size,
                                  epoch=epoch, mask_threshold=self.predictor.mask_threshold,
                                  legacy_revision=identity["resolved_revision"] == "main")
            LOG.info("Model ready on %s, checkpoint=%s", device, self.args.checkpoint)
        except Exception:
            LOG.exception("Could not load model")
            self.predictor = None
            self.categories = []
            messages = {
                "metadata": "Không nạp được danh sách lớp bệnh từ Metadata.csv. "
                            "Kiểm tra đường dẫn --dataset_dir và nội dung metadata trong log terminal.",
                "checkpoint": "Checkpoint không hợp lệ hoặc không tương thích với demo PlantSeg SAM 3. "
                              "Hãy chọn checkpoint khác được tạo bởi pipeline hiện tại.",
                "base_model": "Không nạp được model nền SAM 3. Kiểm tra cache Hugging Face và log terminal. "
                              "Nếu chưa có cache, chạy lại demo với --allow_download "
                              "và tài khoản Hugging Face có quyền truy cập model.",
                "device": "Không đưa được model lên thiết bị xử lý. Kiểm tra bộ nhớ và log terminal; "
                          "có thể chạy lại với --device cpu.",
            }
            with self.state_lock:
                self.state.update(status="error", message=messages[stage])

    def upload_checkpoint(self, stream, length, filename):
        # Never use a browser-supplied filename as a filesystem path.
        filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
        suffix = Path(filename).suffix.lower()
        if suffix not in CHECKPOINT_EXTENSIONS or len(filename) > 255:
            raise RequestError("Hãy chọn checkpoint .pt, .pth hoặc .ckpt.")
        if not 0 < length <= MAX_CHECKPOINT_UPLOAD:
            raise RequestError("Checkpoint phải có dữ liệu và không vượt quá 4 GB.")
        if not self.lock.acquire(blocking=False):
            return 409, {"error": "Model đang bận. Vui lòng đợi rồi thử lại."}
        temporary = None
        try:
            if not self.status()["checkpoint_upload_allowed"]:
                return 409, {"error": "Chỉ tải checkpoint khi model bị thiếu hoặc nạp lỗi."}
            with self.state_lock:
                self.state.update(status="uploading", message="Đang nhận checkpoint…")
            directory = getattr(self.args, "checkpoint_upload_dir", ROOT / "outputs/demo-checkpoints")
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=directory, prefix="checkpoint-", suffix=suffix,
                                             delete=False) as target:
                temporary = Path(target.name)
                remaining = length
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise RequestError("Tải checkpoint bị gián đoạn. Vui lòng thử lại.")
                    target.write(chunk)
                    remaining -= len(chunk)
            previous = self.args.checkpoint if self.uploaded_checkpoint else None
            self.args.checkpoint = temporary
            self.uploaded_checkpoint = True
            with self.state_lock:
                self.state = {"status": "loading", "message": "Đã tải checkpoint. Đang nạp model…",
                              "checkpoint": filename}
            if previous:
                previous.unlink(missing_ok=True)
            # Start after receiving the whole file. load() waits for this request's lock.
            threading.Thread(target=self.load, daemon=True).start()
            return 202, self.status()
        except Exception:
            if temporary:
                temporary.unlink(missing_ok=True)
            with self.state_lock:
                self.state.update(status="error", message="Không tải được checkpoint. Vui lòng thử lại.")
            raise
        finally:
            self.lock.release()

    def sample_list(self):
        # Do not expose labels or disease-bearing filenames before inference.
        return [{"id": i, "name": f"sample-{i + 1:02d}.jpg",
                 "title": f"Ảnh mẫu {i + 1:02d}", "url": f"/api/sample/{i}"}
                for i in range(len(self.samples))]

    def get_sample(self, index):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.samples):
            raise RequestError("Ảnh mẫu không tồn tại.")
        return self.samples[index]

    def sample_image(self, index):
        return decode_image(self.get_sample(index)["image_path"].read_bytes())

    def predict(self, payload):
        if not isinstance(payload, dict):
            raise RequestError("Dữ liệu gửi lên phải là JSON object.")
        if "mode" in payload:
            raise RequestError("Demo dùng một luồng ảnh + prompt. Hãy bỏ trường mode và nhập prompt.")
        prompt = validate_prompt(payload.get("prompt"))
        threshold = payload.get("threshold", getattr(self.predictor, "mask_threshold", 0.5))
        if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not (
            math.isfinite(threshold) and 0.1 <= threshold <= 0.9
        ):
            raise RequestError("Ngưỡng mask phải từ 0.1 đến 0.9.")
        if "sample_id" in payload:
            image = self.sample_image(payload["sample_id"])
        else:
            encoded = payload.get("image")
            if not isinstance(encoded, str):
                raise RequestError("Vui lòng chọn ảnh.")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as error:
                raise RequestError("Ảnh base64 không hợp lệ.") from error
            image = decode_image(raw)
        if self.status()["status"] != "ready":
            return 503, {"error": self.status()["message"]}
        if not self.categories:
            return 503, {"error": "Chưa nạp được danh sách lớp bệnh từ Metadata.csv."}
        if not self.lock.acquire(blocking=False):
            return 409, {"error": "Model đang xử lý một ảnh khác. Vui lòng thử lại sau."}
        try:
            started = time.monotonic()
            validate_prompt_tokens(self.predictor, prompt)
            label = self.reference_index.get(image_fingerprint(image))
            if label is None:
                prediction, classification = predict_disease(
                    self.predictor, image, self.categories, prompt, threshold)
                classification["source"] = "model"
            else:
                category = next((c for c in self.categories if c["name"] == label), None)
                if category is None:
                    raise RuntimeError(f"Reference disease absent from metadata: {label}")
                classification = {
                    "predicted_class": category, "method": "reference_image_lookup",
                    "source": "reference_manifest", "experimental": False,
                    "known_disease_checkpoint": self.predictor.known_disease,
                    "class_count": len(self.categories), "reason": None,
                }
                self.predictor.mask_threshold = threshold
                try:
                    prediction = self.predictor(image, [category])
                finally:
                    self.predictor.text_batches.clear()
            seconds = time.monotonic() - started
            foreground = prediction > 0
            pixels = int(np.count_nonzero(foreground))
            return 200, {"original": png_url(image),
                         "overlay": png_url(mask_overlay(image, foreground, (37, 99, 235))),
                         "mask": png_url(Image.fromarray(foreground.astype(np.uint8) * 255)),
                         "width": image.width, "height": image.height, "pixels": pixels,
                         "coverage": pixels / (image.width * image.height), "seconds": seconds,
                         "prompt": prompt, "threshold": threshold,
                         "classification": classification,
                         "checkpoint": self.status()["checkpoint"]}
        finally:
            self.lock.release()


def make_handler(demo):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, mime="application/json; charset=utf-8", filename=None):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/status":
                self.send(200, demo.status())
            elif path == "/api/samples":
                self.send(200, demo.sample_list())
            elif path == "/api/samples.zip":
                self.send(200, sample_archive(demo.samples_dir, demo.samples),
                          "application/zip", "plantseg-demo-samples.zip")
            elif path.startswith("/api/sample/"):
                try:
                    image = demo.sample_image(int(path.rsplit("/", 1)[1]))
                    stream = io.BytesIO()
                    image.save(stream, "PNG")
                    self.send(200, stream.getvalue(), "image/png")
                except (ValueError, RequestError):
                    self.send(404, {"error": "Ảnh mẫu không tồn tại."})
            elif path in {"/", "/index.html", "/app.js", "/style.css"}:
                name = "index.html" if path == "/" else path[1:]
                mime = {"html": "text/html", "js": "text/javascript", "css": "text/css"}
                self.send(200, (ROOT / "demo" / name).read_bytes(),
                          mime[name.rsplit(".", 1)[1]] + "; charset=utf-8")
            else:
                self.send(404, {"error": "Không tìm thấy nội dung."})

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in {"/api/predict", "/api/checkpoint"}:
                self.send(404, {"error": "Không tìm thấy API."})
                return
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                self.send(403, {"error": "Origin không hợp lệ."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if path == "/api/checkpoint":
                    if not 0 < length <= MAX_CHECKPOINT_UPLOAD:
                        self.close_connection = True
                        self.send(413, {"error": "Checkpoint phải có dữ liệu và không vượt quá 4 GB."})
                        return
                    if self.headers.get("Content-Type") != "application/octet-stream":
                        self.close_connection = True
                        self.send(415, {"error": "Định dạng tải checkpoint không hợp lệ."})
                        return
                    self.connection.settimeout(120)
                    self.close_connection = True
                    code, result = demo.upload_checkpoint(
                        self.rfile, length, unquote(self.headers.get("X-Checkpoint-Name", "")))
                    self.send(code, result)
                    return
                if not 0 < length <= MAX_UPLOAD * 4 // 3 + 4096:
                    self.send(413, {"error": "Yêu cầu quá lớn hoặc không có ảnh (tối đa 12 MB)."})
                    return
                self.connection.settimeout(30)
                payload = json.loads(self.rfile.read(length))
                code, result = demo.predict(payload)
                self.send(code, result)
            except (RequestError, ValueError, UnicodeError) as error:
                self.send(400, {"error": str(error) if isinstance(error, RequestError)
                                 else "Thông tin tải checkpoint không hợp lệ."
                                 if path == "/api/checkpoint" else "Dữ liệu JSON không hợp lệ."})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                LOG.exception("Demo request failed")
                self.send(500, {"error": "Không tải được checkpoint. Kiểm tra dung lượng đĩa và log terminal."
                           if path == "/api/checkpoint" else
                           "Không xử lý được ảnh. Xem log terminal và thử ảnh nhỏ hơn."})

    return Handler


def reload_snapshot(root):
    """Record only source files; model outputs and checkpoints are not watched."""
    paths = [root / "demo.py"]
    paths.extend((root / "src").rglob("*.py"))
    paths.extend(path for path in (root / "demo").rglob("*")
                 if path.suffix in {".html", ".js", ".css"})
    snapshot = {}
    for path in paths:
        try:
            stat = path.stat()
            snapshot[str(path)] = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            # Editors can temporarily remove a file while saving it.
            continue
    return snapshot


def run_reloader(command, root=ROOT):
    """Restart the server subprocess when a watched source file changes."""
    def stop(child):
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    def terminate(_signum, _frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    child = None
    try:
        snapshot = reload_snapshot(root)
        child = subprocess.Popen(command)
        reported_exit = False
        LOG.info("Auto-reload enabled; watching demo.py, src/ and demo/ source files")
        while True:
            time.sleep(1)
            current = reload_snapshot(root)
            if current != snapshot:
                LOG.info("Source changed; restarting demo server")
                stop(child)
                snapshot = current
                child = subprocess.Popen(command)
                reported_exit = False
            elif child.poll() is not None and not reported_exit:
                LOG.warning("Demo server exited (%s); waiting for a source change", child.returncode)
                reported_exit = True
    except KeyboardInterrupt:
        pass
    finally:
        if child is not None:
            stop(child)
        signal.signal(signal.SIGTERM, previous_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/best.pt")
    parser.add_argument("--checkpoint_upload_dir", type=Path, default=ROOT / "outputs/demo-checkpoints",
                        help="Directory for checkpoints uploaded from the demo")
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--cache_dir", type=Path, default=ROOT / "outputs/hf-cache")
    parser.add_argument("--samples_dir", type=Path, default=ROOT / "demo-samples")
    parser.add_argument("--extra_samples_dir", type=Path, default=ROOT / "demo-samples-extra")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allow_download", action="store_true", help="Allow HF base-model downloads")
    parser.add_argument("--reload", action="store_true", help="Restart when demo source files change")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.reload:
        command = [sys.executable, str(Path(__file__).resolve()),
                   *(arg for arg in sys.argv[1:] if arg != "--reload")]
        run_reloader(command)
        return
    args.local_files_only = not args.allow_download
    demo = Demo(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(demo))
    threading.Thread(target=demo.load, daemon=True).start()
    LOG.info("PlantSeg demo: http://%s:%s", args.host, server.server_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
