from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np

# Run this inside the OpenExplorer / OE Docker environment where hrt_model_exec
# is available. It predicts images with the compiled .bin model and the
# bpu_anomaly_stats.npz generated from normal BPU features.
#
# Assumption: images in IMAGE_DIR are already resized/preprocessed to match the
# compiled model runtime input. For the current res_640 model, that means:
# - 640x640
# - RGB image file accepted by hrt_model_exec
# - same resize strategy as the normal feature dump path

MODEL_PATH = Path("model_output/res_640.bin")
STATS_PATH = Path("runs/res_640/bpu_anomaly_stats.npz")
IMAGE_DIR = Path("runs/res_640/images_640_rgb")
OUTPUT_CSV = Path("runs/res_640/docker_predict_result.csv")

HRT_MODEL_EXEC = "hrt_model_exec"
TMP_DIR = Path("tmp/hrt_bpu_predict")
OUTPUT_TXT_NAME = "model_infer_output_0_feature.txt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"model not found: {MODEL_PATH}")
    if not STATS_PATH.exists():
        raise SystemExit(f"stats not found: {STATS_PATH}")

    image_paths = list_images(IMAGE_DIR)
    if not image_paths:
        raise SystemExit(f"no images found: {IMAGE_DIR}")

    center, scale, threshold = load_bpu_stats(STATS_PATH)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"threshold: {threshold:.6f}")
    for index, image_path in enumerate(image_paths):
        raw_feature = dump_one_feature(index, image_path)
        feature = l2_normalize(raw_feature.reshape(1, -1))
        score = score_features(feature, center, scale)[0]
        label = "anomaly" if score > threshold else "normal"
        rows.append((label, float(score), image_path))
        print(f"{label},{score:.6f},{image_path}")

        if (index + 1) % 20 == 0 or index == len(image_paths) - 1:
            print(f"predicted: {index + 1}/{len(image_paths)}")

    save_csv(rows)
    print(f"saved: {OUTPUT_CSV}")


def dump_one_feature(index: int, image_path: Path) -> np.ndarray:
    work_dir = TMP_DIR / f"{index:06d}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    command = [
        HRT_MODEL_EXEC,
        "infer",
        f"--model_file={MODEL_PATH}",
        f"--input_file={image_path}",
        "--enable_dump=true",
        "--dump_format=txt",
        f"--dump_path={work_dir}",
    ]
    subprocess.run(command, check=True)

    output_txt = work_dir / OUTPUT_TXT_NAME
    if not output_txt.exists():
        dumped = "\n".join(str(path) for path in sorted(work_dir.rglob("*")))
        raise FileNotFoundError(
            f"expected output not found: {output_txt}\nDumped files:\n{dumped}"
        )

    feature = np.loadtxt(output_txt, dtype=np.float32).reshape(-1)
    shutil.rmtree(work_dir)
    return feature


def load_bpu_stats(stats_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    data = np.load(stats_path)
    return data["center"], data["scale"], float(data["threshold"])


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, 1e-12)


def score_features(
    features: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    z = (features - center) / scale
    return np.sqrt(np.mean(z * z, axis=1))


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTS else []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def save_csv(rows: list[tuple[str, float, Path]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "score", "path"])
        for label, score, image_path in rows:
            writer.writerow([label, f"{score:.6f}", image_path])


if __name__ == "__main__":
    main()
