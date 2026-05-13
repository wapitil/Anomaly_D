from __future__ import annotations

from pathlib import Path

from core import get_backbone_config, letterbox_resize_rgb, list_images, save_metadata

# Use representative normal images for OpenExplorer post-training quantization.
BACKBONE_NAME = "mobilenet_v2"
GOOD_IMAGE_DIR = Path("/path/to/normal_images")
OUTPUT_DIR = Path("outputs/rdkx5_prefilter/calibration")
MAX_IMAGES = 100


def main() -> None:
    config = get_backbone_config(BACKBONE_NAME)
    image_paths = list_images(GOOD_IMAGE_DIR)[:MAX_IMAGES]
    if not image_paths:
        raise SystemExit(f"no images found: {GOOD_IMAGE_DIR}")

    resized_dir = OUTPUT_DIR / "images_224_rgb"
    resized_dir.mkdir(parents=True, exist_ok=True)
    list_path = OUTPUT_DIR / "calibration_images.txt"

    with list_path.open("w", encoding="utf-8") as file:
        for index, image_path in enumerate(image_paths):
            image = letterbox_resize_rgb(image_path, config.input_size)
            output_path = resized_dir / f"{index:05d}_{image_path.stem}.png"
            image.save(output_path)
            file.write(f"{output_path}\n")

    save_metadata(
        OUTPUT_DIR / "calibration_metadata.json",
        BACKBONE_NAME,
        {
            "image_count": len(image_paths),
            "image_list": str(list_path),
            "purpose": "OpenExplorer PTQ calibration input images.",
        },
    )

    print(f"calibration images: {len(image_paths)}")
    print(f"saved dir: {resized_dir}")
    print(f"saved list: {list_path}")


if __name__ == "__main__":
    main()
