from __future__ import annotations

from pathlib import Path

from core import (
    compute_stats,
    extract_float_features,
    list_images,
    save_metadata,
    save_stats,
)

# Change only this block for your PC-side experiment.
# Available choices:
# - "resnet18"
# - "mobilenet_v2"
# - "mobilenetv4_conv_medium"

BACKBONE_NAME = "mobilenetv4_conv_medium"
GOOD_IMAGE_DIR = Path("Data/jingshu/good/train")
OUTPUT_DIR = Path("runs/mobilenetv4_conv_medium_256")
BATCH_SIZE = 16
THRESHOLD_QUANTILE = 0.995


def main() -> None:
    image_paths = list_images(GOOD_IMAGE_DIR)
    if len(image_paths) < 5:
        raise SystemExit(f"need at least 5 normal images: {GOOD_IMAGE_DIR}")

    features = extract_float_features(image_paths, BACKBONE_NAME, BATCH_SIZE)
    center, scale, threshold, scores = compute_stats(features, THRESHOLD_QUANTILE)

    stats_path = OUTPUT_DIR / "float_anomaly_stats.npz"
    save_stats(
        stats_path,
        center=center,
        scale=scale,
        threshold=threshold,
        backbone_name=BACKBONE_NAME,
        source="pytorch_float",
    )
    save_metadata(
        OUTPUT_DIR / "float_metadata.json",
        BACKBONE_NAME,
        {
            "stats_file": stats_path.name,
            "image_count": len(image_paths),
            "threshold_quantile": THRESHOLD_QUANTILE,
            "feature_postprocess": "l2_normalize_then_zscore_rms",
        },
    )

    print(f"backbone: {BACKBONE_NAME}")
    print(f"normal images: {len(image_paths)}")
    print(f"feature dim: {features.shape[1]}")
    print(f"score mean/max: {scores.mean():.4f}/{scores.max():.4f}")
    print(f"threshold: {threshold:.4f}")
    print(f"saved: {stats_path}")


if __name__ == "__main__":
    main()
