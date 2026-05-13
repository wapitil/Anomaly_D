from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Candidate:
    x: int
    y: int
    w: int
    h: int
    area: int
    score_mean: float
    score_max: float


@dataclass
class TileRecord:
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    source_candidate: int


def odd_kernel(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def normalize_gray(gray: np.ndarray, sigma: float) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    background = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = gray_f - background + 128.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def build_score_map(
    bgr: np.ndarray,
    sigma: float,
    kernel_size: int,
) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    normalized = normalize_gray(gray, sigma)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (odd_kernel(kernel_size), odd_kernel(kernel_size)),
    )

    bright = cv2.morphologyEx(normalized, cv2.MORPH_TOPHAT, kernel)
    dark = cv2.morphologyEx(normalized, cv2.MORPH_BLACKHAT, kernel)
    score = cv2.max(bright, dark)
    score = cv2.GaussianBlur(score, (3, 3), 0)
    return score


def threshold_score(
    score: np.ndarray,
    threshold_percentile: float,
    min_threshold: int,
) -> np.ndarray:
    threshold = max(float(min_threshold), float(np.percentile(score, threshold_percentile)))
    mask = (score >= threshold).astype(np.uint8) * 255
    return mask


def clean_mask(mask: np.ndarray, open_size: int, close_size: int) -> np.ndarray:
    if open_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_kernel(open_size), odd_kernel(open_size)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_kernel(close_size), odd_kernel(close_size)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_candidates(
    mask: np.ndarray,
    score: np.ndarray,
    min_area: int,
) -> list[Candidate]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[Candidate] = []

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        area = int(area)
        if area < min_area:
            continue

        component_scores = score[labels == label]
        candidates.append(
            Candidate(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                area=area,
                score_mean=float(np.mean(component_scores)),
                score_max=float(np.max(component_scores)),
            )
        )

    candidates.sort(key=lambda item: item.area, reverse=True)
    return candidates


def expanded_box(candidate: Candidate, margin: int, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    x1 = max(0, candidate.x - margin)
    y1 = max(0, candidate.y - margin)
    x2 = min(image_w, candidate.x + candidate.w + margin)
    y2 = min(image_h, candidate.y + candidate.h + margin)
    return x1, y1, x2, y2


def boxes_overlap_or_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def merge_candidates(
    candidates: list[Candidate],
    score: np.ndarray,
    image_shape: tuple[int, int, int],
    merge_margin: int,
) -> list[Candidate]:
    if not candidates:
        return []

    image_h, image_w = image_shape[:2]
    boxes = [expanded_box(candidate, merge_margin, image_w, image_h) for candidate in candidates]
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes_overlap_or_touch(boxes[i], boxes[j]):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        groups.setdefault(find(index), []).append(index)

    merged: list[Candidate] = []
    for indices in groups.values():
        x1 = min(candidates[index].x for index in indices)
        y1 = min(candidates[index].y for index in indices)
        x2 = max(candidates[index].x + candidates[index].w for index in indices)
        y2 = max(candidates[index].y + candidates[index].h for index in indices)
        area = sum(candidates[index].area for index in indices)
        region_scores = score[y1:y2, x1:x2]
        merged.append(
            Candidate(
                x=int(x1),
                y=int(y1),
                w=int(x2 - x1),
                h=int(y2 - y1),
                area=int(area),
                score_mean=float(np.mean(region_scores)),
                score_max=float(np.max(region_scores)),
            )
        )

    merged.sort(key=lambda item: item.area, reverse=True)
    return merged


def crop_tile(
    image: np.ndarray,
    candidate: Candidate,
    tile_size: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    cx = candidate.x + candidate.w // 2
    cy = candidate.y + candidate.h // 2
    half = tile_size // 2

    x1 = max(0, min(w - tile_size, cx - half))
    y1 = max(0, min(h - tile_size, cy - half))
    x2 = min(w, x1 + tile_size)
    y2 = min(h, y1 + tile_size)

    tile = image[y1:y2, x1:x2]
    return tile, (x1, y1, x2, y2)


def is_duplicate_tile(
    box: tuple[int, int, int, int],
    existing_boxes: list[tuple[int, int, int, int]],
    min_iou: float,
) -> bool:
    x1, y1, x2, y2 = box
    area = max(0, x2 - x1) * max(0, y2 - y1)
    if area == 0:
        return True

    for other in existing_boxes:
        ox1, oy1, ox2, oy2 = other
        ix1 = max(x1, ox1)
        iy1 = max(y1, oy1)
        ix2 = min(x2, ox2)
        iy2 = min(y2, oy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        other_area = max(0, ox2 - ox1) * max(0, oy2 - oy1)
        union = area + other_area - inter
        if union > 0 and inter / union >= min_iou:
            return True
    return False


def draw_overlay(image: np.ndarray, candidates: list[Candidate]) -> np.ndarray:
    overlay = image.copy()
    for candidate in candidates:
        color = (0, 255, 255)
        x1, y1 = candidate.x, candidate.y
        x2, y2 = candidate.x + candidate.w, candidate.y + candidate.h
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"area={candidate.area}"
        cv2.putText(
            overlay,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def process_image(image_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"skip unreadable image: {image_path}")
        return

    score = build_score_map(
        image,
        sigma=args.sigma,
        kernel_size=args.kernel,
    )
    mask = threshold_score(score, args.threshold_percentile, args.min_threshold)
    mask = clean_mask(mask, args.open_size, args.close_size)
    raw_candidates = find_candidates(mask, score, args.min_area)
    candidates = merge_candidates(raw_candidates, score, image.shape, args.merge_margin)

    stem = image_path.stem
    image_output_dir = output_dir / stem
    tile_dir = image_output_dir / "tiles"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    tile_dir.mkdir(parents=True, exist_ok=True)

    overlay = draw_overlay(image, candidates)
    cv2.imwrite(str(image_output_dir / f"{stem}_overlay.jpg"), overlay)

    tile_records: list[TileRecord] = []
    tile_boxes: list[tuple[int, int, int, int]] = []
    for index, candidate in enumerate(candidates):
        tile, box = crop_tile(image, candidate, args.tile_size)
        if is_duplicate_tile(box, tile_boxes, args.tile_dedupe_iou):
            continue

        tile_boxes.append(box)
        tile_name = f"{stem}_tile_{len(tile_records):03d}_x{box[0]}_y{box[1]}.jpg"
        cv2.imwrite(str(tile_dir / tile_name), tile)
        tile_records.append(
            TileRecord(
                name=tile_name,
                x1=box[0],
                y1=box[1],
                x2=box[2],
                y2=box[3],
                source_candidate=index,
            )
        )

    payload = {
        "image": str(image_path),
        "shape": list(image.shape),
        "parameters": {
            "sigma": args.sigma,
            "kernel": args.kernel,
            "threshold_percentile": args.threshold_percentile,
            "min_threshold": args.min_threshold,
            "min_area": args.min_area,
            "merge_margin": args.merge_margin,
            "tile_size": args.tile_size,
        },
        "raw_candidates_count": len(raw_candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
        "tiles": [asdict(tile) for tile in tile_records],
    }

    with open(image_output_dir / f"{stem}_candidates.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print(
        f"{image_path.name}: raw={len(raw_candidates)} merged={len(candidates)} "
        f"tiles={len(tile_records)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight fabric anomaly candidate prefilter")
    parser.add_argument("--input", required=True, help="input image or image directory")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--sigma", type=float, default=25.0, help="Gaussian sigma for local background removal")
    parser.add_argument("--kernel", type=int, default=41, help="top-hat/black-hat morphology kernel size")
    parser.add_argument("--threshold-percentile", type=float, default=99.4, help="score percentile threshold")
    parser.add_argument("--min-threshold", type=int, default=12, help="absolute minimum score threshold")
    parser.add_argument("--min-area", type=int, default=80, help="ignore connected components smaller than this")
    parser.add_argument("--merge-margin", type=int, default=80, help="merge candidates whose expanded boxes touch")
    parser.add_argument("--open-size", type=int, default=3, help="morphological open kernel size")
    parser.add_argument("--close-size", type=int, default=9, help="morphological close kernel size")
    parser.add_argument("--tile-size", type=int, default=640, help="crop size for candidate tiles")
    parser.add_argument("--tile-dedupe-iou", type=float, default=0.85, help="skip near-duplicate tile crops")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()

    images = list_images(input_path)
    if not images:
        raise SystemExit(f"no images found: {input_path}")

    for image_path in images:
        process_image(image_path, output_dir, args)


if __name__ == "__main__":
    main()
