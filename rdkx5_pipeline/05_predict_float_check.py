from __future__ import annotations

import csv
import sys
from pathlib import Path

from core import extract_float_features, list_images, load_stats, score_features

# This is only a PC float-model check.
# Final RDK prediction should use the BPU model output feature plus bpu_anomaly_stats.npz.
IMAGE_DIR = Path("/path/to/test_images")
STATS_PATH = Path("outputs/rdkx5_prefilter/float_anomaly_stats.npz")
BATCH_SIZE = 32


def main() -> None:
    center, scale, threshold, backbone_name = load_stats(STATS_PATH)
    image_paths = list_images(IMAGE_DIR)
    if not image_paths:
        raise SystemExit(f"no images found: {IMAGE_DIR}")

    features = extract_float_features(image_paths, backbone_name, BATCH_SIZE)
    scores = score_features(features, center, scale)

    writer = csv.writer(sys.stdout)
    writer.writerow(["label", "score", "path"])
    for image_path, score in zip(image_paths, scores):
        label = "anomaly" if score > threshold else "normal"
        writer.writerow([label, f"{score:.4f}", image_path])


if __name__ == "__main__":
    main()
