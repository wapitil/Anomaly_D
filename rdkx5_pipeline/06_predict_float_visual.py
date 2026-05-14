from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from core import extract_float_features, list_images, load_stats, score_features
from PIL import Image, ImageDraw, ImageFont

# Run 01_train_float_stats.py first, then change this block and run this file.
NAME = "jietou"
IMAGE_DIR = Path("Data/jingshu") / NAME
PROJECT = "res_640"
STATS_PATH = Path("runs") / PROJECT / "float_anomaly_stats.npz"
OUTPUT_DIR = Path("runs") / PROJECT / f"{NAME}_visual"
BATCH_SIZE = 32


def main() -> None:
    center, scale, threshold, backbone_name = load_stats(STATS_PATH)
    image_paths = list_images(IMAGE_DIR)
    if not image_paths:
        raise SystemExit(f"no images found: {IMAGE_DIR}")

    features = extract_float_features(image_paths, backbone_name, BATCH_SIZE)
    scores = score_features(features, center, scale)

    rows = []
    for image_path, score in zip(image_paths, scores):
        label = "anomaly" if score > threshold else "normal"
        rows.append((label, float(score), image_path))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # csv_path = OUTPUT_DIR / "pc_predict_result.csv"
    visual_path = OUTPUT_DIR / "pc_predict_summary.jpg"

    # save_csv(csv_path, rows)
    save_visual_summary(visual_path, rows, threshold)

    anomaly_count = sum(1 for label, _, _ in rows if label == "anomaly")
    print(f"stats: {STATS_PATH}")
    print(f"test images: {len(rows)}")
    print(f"normal: {len(rows) - anomaly_count}")
    print(f"anomaly: {anomaly_count}")
    # print(f"saved csv: {csv_path}")
    print(f"saved visual: {visual_path}")


def save_csv(csv_path: Path, rows: list[tuple[str, float, Path]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "score", "path"])
        for label, score, image_path in rows:
            writer.writerow([label, f"{score:.4f}", image_path])


def save_visual_summary(
    output_path: Path, rows: list[tuple[str, float, Path]], threshold: float
) -> None:
    rows = sorted(rows, reverse=True, key=lambda item: item[1])

    tile_w = 300
    image_h = 260
    header_h = 48
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
        text_color = (255, 255, 255)

        draw.rectangle((x, y, x + tile_w, y + header_h), fill=header_color)
        draw.text(
            (x + 10, y + 8),
            f"{label.upper()}  score={score:.4f}  th={threshold:.4f}",
            fill=text_color,
            font=font,
        )
        draw.text(
            (x + 10, y + 27),
            image_path.name[:42],
            fill=text_color,
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
                (x + 16, y + header_h + 116),
                "read failed",
                fill=header_color,
                font=font,
            )

    sheet.save(output_path)


if __name__ == "__main__":
    main()
