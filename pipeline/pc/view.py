from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# PC: ONNX sanity visualization.
# It fits a simple normal center from NORMAL_DIR with the ONNX feature model,
# then scores IMAGE_DIR and writes predict.csv + summary.jpg.
# This is only a PC-side trend check. Final thresholds must be fitted on RDK.

ONNX_PATH = Path("runs/res_640/onnx/res_640.onnx")
NORMAL_DIR = Path("Data/jingshu/good/train")
IMAGE_DIR = Path("Data/jingshu/test")
OUTPUT_DIR = Path("runs/res_640/onnx_view")
INPUT_SIZE = 640
THRESHOLD_SCALE = 1.8
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def main() -> None:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime is required: pip install onnxruntime") from exc

    if not ONNX_PATH.exists():
        raise SystemExit(f"onnx not found: {ONNX_PATH}")

    normal_paths = list_images(NORMAL_DIR)
    image_paths = list_images(IMAGE_DIR)
    if len(normal_paths) < 2:
        raise SystemExit(f"need at least 2 normal images: {NORMAL_DIR}")
    if not image_paths:
        raise SystemExit(f"no images found: {IMAGE_DIR}")

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    normal_features = extract_features(session, input_name, normal_paths)
    center = normal_features.mean(axis=0).astype(np.float32)
    normal_scores = np.linalg.norm(normal_features - center, axis=1)
    threshold = float(normal_scores.max() * THRESHOLD_SCALE)

    features = extract_features(session, input_name, image_paths)
    scores = np.linalg.norm(features - center, axis=1)
    rows = []
    for image_path, score in zip(image_paths, scores):
        label = "anomaly" if score > threshold else "normal"
        rows.append((label, float(score), image_path))
        print(f"{label},{score:.6f},{image_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(OUTPUT_DIR / "predict.csv", rows)
    save_visual(OUTPUT_DIR / "summary.jpg", rows, threshold)
    print(f"threshold: {threshold:.6f}")
    print(f"saved: {OUTPUT_DIR}")


def extract_features(session: object, input_name: str, image_paths: list[Path]) -> np.ndarray:
    features = []
    for image_path in image_paths:
        tensor = preprocess(image_path)
        output = session.run(None, {input_name: tensor})[0]
        feature = output.reshape(1, -1).astype(np.float32)
        features.append(l2_normalize(feature)[0])
    return np.stack(features).astype(np.float32)


def preprocess(image_path: Path) -> np.ndarray:
    image = letterbox_resize_rgb(image_path, INPUT_SIZE)
    array = np.asarray(image).astype(np.float32) / 255.0
    array = (array - MEAN) / STD
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0).astype(np.float32)


def letterbox_resize_rgb(image_path: Path, input_size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
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


def save_visual(output_path: Path, rows: list[tuple[str, float, Path]], threshold: float) -> None:
    rows = sorted(rows, reverse=True, key=lambda item: item[1])
    tile_w, image_h, header_h, gap, margin, cols = 300, 240, 52, 12, 16, 4
    row_count = max(1, int(np.ceil(len(rows) / cols)))
    sheet = Image.new(
        "RGB",
        (margin * 2 + cols * tile_w + (cols - 1) * gap, margin * 2 + row_count * (header_h + image_h + gap) - gap),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (label, score, image_path) in enumerate(rows):
        x = margin + (index % cols) * (tile_w + gap)
        y = margin + (index // cols) * (header_h + image_h + gap)
        color = (205, 45, 40) if label == "anomaly" else (40, 140, 75)
        draw.rectangle((x, y, x + tile_w, y + header_h), fill=color)
        draw.text((x + 8, y + 7), f"{label.upper()}  score={score:.4f}", fill=(255, 255, 255), font=font)
        draw.text((x + 8, y + 27), f"th={threshold:.4f}  {image_path.name[:30]}", fill=(255, 255, 255), font=font)
        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((tile_w, image_h))
            sheet.paste(image, (x + (tile_w - image.width) // 2, y + header_h + (image_h - image.height) // 2))
        except Exception:
            draw.text((x + 16, y + header_h + 108), "read failed", fill=color, font=font)
    sheet.save(output_path)


if __name__ == "__main__":
    main()
