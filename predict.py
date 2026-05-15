from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# RDK web/runtime prediction backend.
# One model path, one stats path, one preprocessing path.
# fit_stats() and predict_image() intentionally share the same BPU feature code.

MODEL_PATH = Path("pipeline/models/res_640.bin")
STATS_PATH = Path("pipeline/stats/bpu_anomaly_stats.npz")
NORMAL_DIR = Path("pipeline/normal_images")
INPUT_SIZE = 640
THRESHOLD_QUANTILE = 0.995
THRESHOLD_SCALE = 1.8
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = BpuPredictor(MODEL_PATH, STATS_PATH)
    return _predictor


def reload_model() -> None:
    global _predictor
    _predictor = None
    print("BPU predictor will reload on next frame")


def predict_image(image_bgr: np.ndarray) -> dict[str, object]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("empty image for prediction")
    return get_predictor().predict_image(image_bgr)


def fit_stats(normal_dir: Path = NORMAL_DIR) -> None:
    normal_paths = list_images(normal_dir)
    if len(normal_paths) < 2:
        raise RuntimeError(f"need at least 2 normal images: {normal_dir}")

    predictor = BpuPredictor(MODEL_PATH, None)
    features = []
    for image_path in normal_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        raw_feature = predictor.extract_feature(image)
        features.append(l2_normalize(raw_feature.reshape(1, -1))[0])

    if len(features) < 2:
        raise RuntimeError(f"not enough readable normal images: {normal_dir}")

    features_array = np.stack(features).astype(np.float32)
    center = features_array.mean(axis=0).astype(np.float32)
    scale = np.ones_like(center, dtype=np.float32)
    scores = np.linalg.norm(features_array - center, axis=1)
    threshold = float(np.quantile(scores, THRESHOLD_QUANTILE) * THRESHOLD_SCALE)

    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        STATS_PATH,
        center=center,
        scale=scale,
        threshold=np.array(threshold, dtype=np.float32),
        backbone=np.array("resnet18"),
        source=np.array("rdk_bpu_infer_lib"),
        score_method=np.array("l2_center"),
    )
    reload_model()
    print(f"fit stats saved: {STATS_PATH}, threshold={threshold:.6f}, images={len(features)}")


class BpuPredictor:
    def __init__(self, model_path: Path, stats_path: Path | None) -> None:
        try:
            import bpu_infer_lib
        except ImportError as exc:
            raise RuntimeError("bpu_infer_lib is required on RDK X5") from exc

        if not model_path.exists():
            raise RuntimeError(f"model not found: {model_path}")

        self.infer = bpu_infer_lib.Infer(False)
        self.infer.load_model(str(model_path))
        self.center = None
        self.scale = None
        self.threshold = None
        self.score_method = "l2_center"

        if stats_path is not None:
            self.load_stats(stats_path)

    def load_stats(self, stats_path: Path) -> None:
        if not stats_path.exists():
            raise RuntimeError(f"stats not found: {stats_path}")
        data = np.load(stats_path)
        self.center = data["center"].astype(np.float32)
        self.scale = data["scale"].astype(np.float32)
        self.threshold = float(data["threshold"])
        self.score_method = str(data["score_method"]) if "score_method" in data.files else "zscore"
        print(f"loaded stats: {stats_path}, threshold={self.threshold:.6f}, method={self.score_method}")

    def predict_image(self, image_bgr: np.ndarray) -> dict[str, object]:
        raw_feature = self.extract_feature(image_bgr)
        feature = l2_normalize(raw_feature.reshape(1, -1))
        score = float(score_feature(feature, self.center, self.scale, self.score_method)[0])
        pred_label = score > self.threshold
        result = draw_result(image_bgr, score, self.threshold, pred_label)
        return {
            "score": score,
            "pred_label": pred_label,
            "result": result,
        }

    def extract_feature(self, image_bgr: np.ndarray) -> np.ndarray:
        input_tensor = preprocess_image(image_bgr)
        self.infer.read_numpy_arr_uint8(input_tensor, 0)
        self.infer.forward()
        return np.asarray(self.infer.get_infer_res_np_float32(0)).reshape(-1).astype(np.float32)


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)
    image = letterbox_resize_rgb(image, INPUT_SIZE)
    array = np.asarray(image).astype(np.uint8)
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0)


def letterbox_resize_rgb(image: Image.Image, input_size: int) -> Image.Image:
    width, height = image.size
    scale = min(input_size / width, input_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = image.resize((resized_width, resized_height), RESAMPLE_BILINEAR)
    canvas = Image.new("RGB", (input_size, input_size), (0, 0, 0))
    canvas.paste(resized, ((input_size - resized_width) // 2, (input_size - resized_height) // 2))
    return canvas


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, 1e-12)


def score_feature(features: np.ndarray, center: np.ndarray, scale: np.ndarray, score_method: str) -> np.ndarray:
    if score_method == "l2_center":
        return np.linalg.norm(features - center, axis=1)
    if score_method == "zscore":
        z = (features - center) / scale
        return np.sqrt(np.mean(z * z, axis=1))
    raise ValueError(f"unsupported score method: {score_method}")


def draw_result(image_bgr: np.ndarray, score: float, threshold: float, pred_label: bool) -> np.ndarray:
    result = image_bgr.copy()
    color = (0, 0, 255) if pred_label else (0, 180, 0)
    label = "ANOMALY" if pred_label else "NORMAL"
    h, w = result.shape[:2]
    cv2.rectangle(result, (0, 0), (w - 1, h - 1), color, 8)
    cv2.putText(
        result,
        f"{label}  score={score:.4f}  th={threshold:.4f}",
        (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        color,
        4,
        cv2.LINE_AA,
    )
    return result


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTS else []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def image_stream():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("camera open failed")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        while True:
            ret, image = cap.read()
            if not ret or image is None:
                time.sleep(0.03)
                continue
            yield image
    finally:
        cap.release()
