from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

# Run this inside the OpenExplorer / OE Docker environment where hrt_model_exec
# is available. This script intentionally avoids importing project modules such
# as core.py, because the Docker image may not have torch/torchvision/PIL.
#
# Assumption: images in IMAGE_DIR are already resized/preprocessed to match the
# compiled model runtime input. For the current res_640 model, that means:
# - 640x640
# - RGB image file accepted by hrt_model_exec
# - same resize strategy as the model calibration/export path

MODEL_PATH = Path("model_output/res_640.bin")
IMAGE_DIR = Path("runs/res_640/calibration")
OUTPUT_FEATURE_FILE = Path("runs/res_640/bpu_normal_features.npy")

HRT_MODEL_EXEC = "hrt_model_exec"
TMP_DIR = Path("tmp/hrt_bpu_feature_dump")
OUTPUT_TXT_NAME = "model_infer_output_0_feature.txt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"model not found: {MODEL_PATH}")

    image_paths = list_images(IMAGE_DIR)
    if not image_paths:
        raise SystemExit(f"no images found: {IMAGE_DIR}")

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    features: list[np.ndarray] = []

    for index, image_path in enumerate(image_paths):
        feature = dump_one_feature(index, image_path)
        features.append(feature)
        if (index + 1) % 20 == 0 or index == len(image_paths) - 1:
            print(f"dumped: {index + 1}/{len(image_paths)}")

    feature_array = np.stack(features, axis=0).astype(np.float32)
    OUTPUT_FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_FEATURE_FILE, feature_array)

    print(f"feature shape: {feature_array.shape}")
    print(f"saved: {OUTPUT_FEATURE_FILE}")


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


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTS else []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


if __name__ == "__main__":
    main()
