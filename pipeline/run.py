from __future__ import annotations

import csv
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# RDK: standalone final prediction script.
# It does not import any project helper code.
# Input: .bin model, bpu_anomaly_stats.npz, and images to predict.
# Output: prediction CSV and a visual summary image.

MODEL_PATH = Path("pipeline/models/res_640.bin")
STATS_PATH = Path("pipeline/stats/bpu_anomaly_stats.npz")
IMAGE_DIR = Path("collected_data/Images_70")
OUTPUT_DIR = Path("pipeline/outputs/rdk_predict")

INPUT_SIZE = 640
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def main() -> None:
    args = parse_args()

    try:
        import bpu_infer_lib
    except ImportError as exc:
        raise SystemExit(
            "bpu_infer_lib import failed.\n"
            "This script must run with the RDK X5 Python environment that has "
            "bpu_infer_lib_x5 installed.\n"
            f"python: {sys.executable}\n"
            f"version: {sys.version}\n"
            "Try:\n"
            "  python3 -m pip install bpu_infer_lib_x5 "
            "-i http://sdk.d-robotics.cc:8080/simple/ "
            "--trusted-host sdk.d-robotics.cc\n"
            "Then verify:\n"
            "  python3 -c \"import bpu_infer_lib; print('ok')\""
        ) from exc

    if not args.model_path.exists():
        raise SystemExit(f"model not found: {args.model_path}")
    if not args.stats_path.exists():
        raise SystemExit(f"stats not found: {args.stats_path}")

    image_paths = list_images(args.image_dir)
    if not image_paths:
        raise SystemExit(f"no images found: {args.image_dir}")

    center, scale, threshold, score_method = load_stats(args.stats_path)

    infer = bpu_infer_lib.Infer(False)
    infer.load_model(str(args.model_path))
    print_model_info(infer, args.model_path, args.stats_path, args.image_dir)
    print(f"threshold: {threshold:.6f}")

    rows: list[tuple[str, float, Path]] = []
    for index, image_path in enumerate(image_paths):
        raw_feature = run_bpu_feature(infer, image_path)
        feature = l2_normalize(raw_feature.reshape(1, -1))
        score = score_feature(feature, center, scale, score_method)[0]
        label = "anomaly" if score > threshold else "normal"
        rows.append((label, float(score), image_path))
        print(f"{label},{score:.6f},{image_path}")

        if (index + 1) % 20 == 0 or index == len(image_paths) - 1:
            print(f"predicted: {index + 1}/{len(image_paths)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "predict.csv"
    visual_path = args.output_dir / "summary.jpg"

    save_csv(csv_path, rows)
    save_visual_summary(visual_path, rows, threshold)

    print(f"saved csv: {csv_path}")
    print(f"saved visual: {visual_path}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--stats-path", type=Path, default=STATS_PATH)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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


def load_stats(stats_path: Path) -> tuple[np.ndarray, np.ndarray, float, str]:
    data = np.load(stats_path)
    score_method = str(data["score_method"]) if "score_method" in data.files else "zscore"
    return data["center"], data["scale"], float(data["threshold"]), score_method


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, 1e-12)


def score_feature(
    features: np.ndarray, center: np.ndarray, scale: np.ndarray, score_method: str
) -> np.ndarray:
    if score_method == "l2_center":
        return np.linalg.norm(features - center, axis=1)
    if score_method == "zscore":
        z = (features - center) / scale
        return np.sqrt(np.mean(z * z, axis=1))
    raise ValueError(f"unsupported score method: {score_method}")


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTS else []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def save_csv(csv_path: Path, rows: list[tuple[str, float, Path]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "score", "path"])
        for label, score, image_path in rows:
            writer.writerow([label, f"{score:.6f}", image_path])


def save_visual_summary(
    output_path: Path, rows: list[tuple[str, float, Path]], threshold: float
) -> None:
    rows = sorted(rows, reverse=True, key=lambda item: item[1])

    tile_w = 300
    image_h = 240
    header_h = 52
    gap = 12
    margin = 16
    cols = 4
    row_count = max(1, int(np.ceil(len(rows) / cols)))
    width = margin * 2 + cols * tile_w + (cols - 1) * gap
    height = margin * 2 + row_count * (header_h + image_h + gap) - gap

    sheet = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, (label, score, image_path) in enumerate(rows):
        x = margin + (index % cols) * (tile_w + gap)
        y = margin + (index // cols) * (header_h + image_h + gap)
        is_anomaly = label == "anomaly"
        header_color = (205, 45, 40) if is_anomaly else (40, 140, 75)

        draw.rectangle((x, y, x + tile_w, y + header_h), fill=header_color)
        draw.text(
            (x + 8, y + 7),
            f"{label.upper()}  score={score:.4f}",
            fill=(255, 255, 255),
            font=font,
        )
        draw.text(
            (x + 8, y + 27),
            f"th={threshold:.4f}  {image_path.name[:30]}",
            fill=(255, 255, 255),
            font=font,
        )

        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((tile_w, image_h))
            px = x + (tile_w - image.width) // 2
            py = y + header_h + (image_h - image.height) // 2
            sheet.paste(image, (px, py))
        except Exception:
            draw.rectangle(
                (x, y + header_h, x + tile_w, y + header_h + image_h),
                outline=header_color,
                width=2,
            )
            draw.text(
                (x + 16, y + header_h + 108),
                "read failed",
                fill=header_color,
                font=font,
            )

    sheet.save(output_path)


def print_model_info(
    infer: object, model_path: Path, stats_path: Path, image_dir: Path
) -> None:
    print(f"model: {model_path}")
    print(f"stats: {stats_path}")
    print(f"image dir: {image_dir}")
    print(f"inputs: {len(infer.inputs)}")
    for index, tensor in enumerate(infer.inputs):
        props = tensor.properties
        shape = get_tensor_shape(tensor, props)
        print(
            f"input[{index}]: layout={props.tensorLayout}, "
            f"type={props.tensorType}, shape={shape}"
        )

    print(f"outputs: {len(infer.outputs)}")
    for index, tensor in enumerate(infer.outputs):
        props = tensor.properties
        shape = get_tensor_shape(tensor, props)
        print(
            f"output[{index}]: layout={props.tensorLayout}, "
            f"type={props.tensorType}, shape={shape}"
        )


def get_tensor_shape(tensor: object, props: object) -> object:
    data = getattr(tensor, "data", None)
    if data is not None and hasattr(data, "shape"):
        return data.shape

    for attr in ("validShape", "alignedShape", "shape"):
        value = getattr(props, attr, None)
        if value is not None:
            return value

    return "unknown"


if __name__ == "__main__":
    main()
