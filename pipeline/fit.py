from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image

# RDK: standalone stats builder.
# Use this on the same RDK runtime path as final prediction.
# Input: .bin model and normal images.
# Output: bpu_anomaly_stats.npz for classify.py.

MODEL_PATH = Path("pipeline/models/res_640.bin")
NORMAL_IMAGE_DIR = Path("pipeline/normal_images")
OUTPUT_STATS_PATH = Path("pipeline/stats/bpu_anomaly_stats.npz")
OUTPUT_METADATA_PATH = Path("pipeline/stats/bpu_metadata.json")

INPUT_SIZE = 640
THRESHOLD_QUANTILE = 0.995
THRESHOLD_SCALE = 1.8
SCORE_METHOD = "l2_center"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def main() -> None:
    args = parse_args()

    try:
        import bpu_infer_lib
    except ImportError as exc:
        raise SystemExit(
            "bpu_infer_lib import failed.\n"
            f"python: {sys.executable}\n"
            "Install bpu_infer_lib_x5 in this RDK Python environment."
        ) from exc

    if not args.model_path.exists():
        raise SystemExit(f"model not found: {args.model_path}")

    image_paths = list_images(args.normal_dir)
    if len(image_paths) < 2:
        raise SystemExit(f"need at least 2 normal images: {args.normal_dir}")

    infer = bpu_infer_lib.Infer(False)
    infer.load_model(str(args.model_path))

    features = []
    for index, image_path in enumerate(image_paths):
        raw_feature = run_bpu_feature(infer, image_path)
        feature = l2_normalize(raw_feature.reshape(1, -1))[0]
        features.append(feature)

        if (index + 1) % 20 == 0 or index == len(image_paths) - 1:
            print(f"processed: {index + 1}/{len(image_paths)}")

    features_array = np.stack(features).astype(np.float32)
    center, scale, threshold, scores = fit_stats(
        features_array,
        args.threshold_quantile,
        args.threshold_scale,
        args.score_method,
    )

    args.output_stats.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_stats,
        center=center,
        scale=scale,
        threshold=np.array(threshold, dtype=np.float32),
        backbone=np.array("resnet18"),
        source=np.array("rdk_bpu_infer_lib"),
        score_method=np.array(args.score_method),
    )
    args.output_metadata.write_text(
        json.dumps(
            {
                "model": str(args.model_path),
                "normal_image_dir": str(args.normal_dir),
                "normal_image_count": len(image_paths),
                "feature_shape": list(features_array.shape),
                "threshold_quantile": args.threshold_quantile,
                "threshold_scale": args.threshold_scale,
                "threshold": threshold,
                "score_mean": float(scores.mean()),
                "score_max": float(scores.max()),
                "score_method": args.score_method,
                "preprocess": "letterbox_resize_rgb_uint8_nchw",
                "source": "rdk_bpu_infer_lib",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"features: {features_array.shape}")
    print(f"score mean/max: {scores.mean():.6f}/{scores.max():.6f}")
    print(f"threshold: {threshold:.6f}")
    print(f"saved stats: {args.output_stats}")
    print(f"saved metadata: {args.output_metadata}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--normal-dir", type=Path, default=NORMAL_IMAGE_DIR)
    parser.add_argument("--output-stats", type=Path, default=OUTPUT_STATS_PATH)
    parser.add_argument("--output-metadata", type=Path, default=OUTPUT_METADATA_PATH)
    parser.add_argument("--threshold-quantile", type=float, default=THRESHOLD_QUANTILE)
    parser.add_argument("--threshold-scale", type=float, default=THRESHOLD_SCALE)
    parser.add_argument(
        "--score-method",
        choices=("l2_center", "zscore"),
        default=SCORE_METHOD,
    )
    return parser.parse_args()


def run_bpu_feature(infer: object, image_path: Path) -> np.ndarray:
    input_tensor = preprocess_image(image_path)
    infer.read_numpy_arr_uint8(input_tensor, 0)
    infer.forward()
    return np.asarray(infer.get_infer_res_np_float32(0)).reshape(-1).astype(np.float32)


def preprocess_image(image_path: Path) -> np.ndarray:
    image = letterbox_resize_rgb(image_path, INPUT_SIZE)
    array = np.asarray(image).astype(np.uint8)
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0)


def letterbox_resize_rgb(image_path: Path, input_size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    scale = min(input_size / width, input_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = image.resize((resized_width, resized_height), RESAMPLE_BILINEAR)

    canvas = Image.new("RGB", (input_size, input_size), (0, 0, 0))
    left = (input_size - resized_width) // 2
    top = (input_size - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, 1e-12)


def fit_stats(
    features: np.ndarray,
    threshold_quantile: float,
    threshold_scale: float,
    score_method: str,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    center = features.mean(axis=0).astype(np.float32)
    if score_method == "l2_center":
        scale = np.ones_like(center, dtype=np.float32)
        scores = np.linalg.norm(features - center, axis=1)
    elif score_method == "zscore":
        scale = (features.std(axis=0) + 1e-6).astype(np.float32)
        scores = score_features(features, center, scale)
    else:
        raise ValueError(f"unsupported score method: {score_method}")

    threshold = float(np.quantile(scores, threshold_quantile) * threshold_scale)
    return center, scale, threshold, scores


def score_features(
    features: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    z = (features - center) / scale
    return np.sqrt(np.mean(z * z, axis=1))


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTS else []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


if __name__ == "__main__":
    main()
