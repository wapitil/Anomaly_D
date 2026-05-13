from __future__ import annotations

from pathlib import Path

from core import compute_stats, l2_normalize, load_feature_file, save_metadata, save_stats

# Run this after ONNX -> OpenExplorer -> .hbm/.bin.
# First use the quantized BPU model to run all normal images and save raw features.
# Then point BPU_FEATURE_FILE to that exported feature file.
BACKBONE_NAME = "mobilenet_v2"
BPU_FEATURE_FILE = Path("outputs/rdkx5_prefilter/bpu_normal_features.npy")
OUTPUT_DIR = Path("outputs/rdkx5_prefilter")
THRESHOLD_QUANTILE = 0.995

# Keep True when the BPU output is the raw backbone feature.
# Set False only if your RDK-side feature dump has already done L2 normalization.
BPU_FEATURE_IS_RAW = True


def main() -> None:
    features = load_feature_file(BPU_FEATURE_FILE)
    if BPU_FEATURE_IS_RAW:
        features = l2_normalize(features)

    center, scale, threshold, scores = compute_stats(features, THRESHOLD_QUANTILE)
    stats_path = OUTPUT_DIR / "bpu_anomaly_stats.npz"
    save_stats(
        stats_path,
        center=center,
        scale=scale,
        threshold=threshold,
        backbone_name=BACKBONE_NAME,
        source="rdkx5_bpu_quantized",
    )
    save_metadata(
        OUTPUT_DIR / "bpu_metadata.json",
        BACKBONE_NAME,
        {
            "stats_file": stats_path.name,
            "feature_file": str(BPU_FEATURE_FILE),
            "feature_count": features.shape[0],
            "feature_dim": features.shape[1],
            "threshold_quantile": THRESHOLD_QUANTILE,
            "feature_postprocess": "l2_normalize_then_zscore_rms",
        },
    )

    print(f"features: {features.shape}")
    print(f"score mean/max: {scores.mean():.4f}/{scores.max():.4f}")
    print(f"threshold: {threshold:.4f}")
    print(f"saved: {stats_path}")


if __name__ == "__main__":
    main()
