from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SLICE_SIZE = 1280
SIGMA = 25.0
KERNEL_SIZE = 41
THRESHOLD_PERCENTILE = 99.4
MIN_THRESHOLD = 12
MIN_AREA = 80
MERGE_MARGIN = 80
OPEN_SIZE = 3
CLOSE_SIZE = 9
TILE_DEDUPE_IOU = 0.85


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
class SliceRecord:
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


def normalize_gray(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    background = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=SIGMA, sigmaY=SIGMA)
    normalized = gray_f - background + 128.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def build_score_map(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    normalized = normalize_gray(gray)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (odd_kernel(KERNEL_SIZE), odd_kernel(KERNEL_SIZE)),
    )

    bright = cv2.morphologyEx(normalized, cv2.MORPH_TOPHAT, kernel)
    dark = cv2.morphologyEx(normalized, cv2.MORPH_BLACKHAT, kernel)
    score = cv2.max(bright, dark)
    score = cv2.GaussianBlur(score, (3, 3), 0)
    return score


def threshold_score(score: np.ndarray) -> np.ndarray:
    threshold = max(
        float(MIN_THRESHOLD), float(np.percentile(score, THRESHOLD_PERCENTILE))
    )
    return (score >= threshold).astype(np.uint8) * 255


def clean_mask(mask: np.ndarray) -> np.ndarray:
    if OPEN_SIZE > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (odd_kernel(OPEN_SIZE), odd_kernel(OPEN_SIZE))
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if CLOSE_SIZE > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (odd_kernel(CLOSE_SIZE), odd_kernel(CLOSE_SIZE))
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_candidates(
    mask: np.ndarray, score: np.ndarray, offset_x: int, offset_y: int
) -> list[Candidate]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    candidates: list[Candidate] = []

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        area = int(area)
        if area < MIN_AREA:
            continue

        component_scores = score[labels == label]
        candidates.append(
            Candidate(
                x=int(x + offset_x),
                y=int(y + offset_y),
                w=int(w),
                h=int(h),
                area=area,
                score_mean=float(np.mean(component_scores)),
                score_max=float(np.max(component_scores)),
            )
        )

    candidates.sort(key=lambda item: item.area, reverse=True)
    return candidates


def slice_origins(length: int, slice_size: int = SLICE_SIZE) -> list[int]:
    if length <= slice_size:
        return [0]

    last_origin = length - slice_size
    origins = list(range(0, last_origin + 1, slice_size))
    if origins[-1] != last_origin:
        origins.append(last_origin)
    return origins


def iter_slices(
    image: np.ndarray,
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    h, w = image.shape[:2]
    slices = []
    for y1 in slice_origins(h):
        for x1 in slice_origins(w):
            x2 = min(w, x1 + SLICE_SIZE)
            y2 = min(h, y1 + SLICE_SIZE)
            slices.append((image[y1:y2, x1:x2], (x1, y1, x2, y2)))
    return slices


def detect_candidates_in_slices(image: np.ndarray) -> tuple[list[Candidate], int]:
    all_candidates: list[Candidate] = []
    slices = iter_slices(image)

    for image_slice, (x1, y1, _, _) in slices:
        score = build_score_map(image_slice)
        mask = threshold_score(score)
        mask = clean_mask(mask)
        all_candidates.extend(find_candidates(mask, score, x1, y1))

    all_candidates.sort(key=lambda item: item.area, reverse=True)
    return all_candidates, len(slices)


def expanded_box(
    candidate: Candidate, margin: int, image_w: int, image_h: int
) -> tuple[int, int, int, int]:
    x1 = max(0, candidate.x - margin)
    y1 = max(0, candidate.y - margin)
    x2 = min(image_w, candidate.x + candidate.w + margin)
    y2 = min(image_h, candidate.y + candidate.h + margin)
    return x1, y1, x2, y2


def boxes_overlap_or_touch(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def merge_candidates(
    candidates: list[Candidate], image_shape: tuple[int, int, int]
) -> list[Candidate]:
    if not candidates:
        return []

    image_h, image_w = image_shape[:2]
    boxes = [
        expanded_box(candidate, MERGE_MARGIN, image_w, image_h)
        for candidate in candidates
    ]
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
        score_mean = float(np.mean([candidates[index].score_mean for index in indices]))
        score_max = float(max(candidates[index].score_max for index in indices))
        merged.append(
            Candidate(
                x=int(x1),
                y=int(y1),
                w=int(x2 - x1),
                h=int(y2 - y1),
                area=int(area),
                score_mean=score_mean,
                score_max=score_max,
            )
        )

    merged.sort(key=lambda item: item.area, reverse=True)
    return merged


def candidate_slice_box(
    candidate: Candidate, image_w: int, image_h: int
) -> tuple[int, int, int, int]:
    cx = candidate.x + candidate.w // 2
    cy = candidate.y + candidate.h // 2
    half = SLICE_SIZE // 2

    if image_w <= SLICE_SIZE:
        x1 = 0
    else:
        x1 = max(0, min(image_w - SLICE_SIZE, cx - half))

    if image_h <= SLICE_SIZE:
        y1 = 0
    else:
        y1 = max(0, min(image_h - SLICE_SIZE, cy - half))

    x2 = min(image_w, x1 + SLICE_SIZE)
    y2 = min(image_h, y1 + SLICE_SIZE)
    return x1, y1, x2, y2


def is_duplicate_box(
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


def process_image(image_path: Path, output_dir: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"skip unreadable image: {image_path}")
        return

    raw_candidates, slice_count = detect_candidates_in_slices(image)
    candidates = merge_candidates(raw_candidates, image.shape)

    stem = image_path.stem
    image_output_dir = output_dir / stem
    slice_dir = image_output_dir / "slices"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    slice_dir.mkdir(parents=True, exist_ok=True)

    slice_records: list[SliceRecord] = []
    slice_boxes: list[tuple[int, int, int, int]] = []
    image_h, image_w = image.shape[:2]
    for index, candidate in enumerate(candidates):
        box = candidate_slice_box(candidate, image_w, image_h)
        if is_duplicate_box(box, slice_boxes, TILE_DEDUPE_IOU):
            continue

        x1, y1, x2, y2 = box
        image_slice = image[y1:y2, x1:x2]
        slice_boxes.append(box)
        slice_name = f"{stem}_slice_{len(slice_records):03d}_x{x1}_y{y1}.jpg"
        cv2.imwrite(str(slice_dir / slice_name), image_slice)
        slice_records.append(
            SliceRecord(
                name=slice_name,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                source_candidate=index,
            )
        )

    payload = {
        "image": str(image_path),
        "shape": list(image.shape),
        "parameters": {
            "slice_size": SLICE_SIZE,
            "sigma": SIGMA,
            "kernel": KERNEL_SIZE,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "min_threshold": MIN_THRESHOLD,
            "min_area": MIN_AREA,
            "merge_margin": MERGE_MARGIN,
            "open_size": OPEN_SIZE,
            "close_size": CLOSE_SIZE,
            "tile_dedupe_iou": TILE_DEDUPE_IOU,
        },
        "slice_count": slice_count,
        "raw_candidates_count": len(raw_candidates),
        "merged_candidate_count": len(candidates),
        "slices": [asdict(image_slice) for image_slice in slice_records],
    }

    with open(
        image_output_dir / f"{stem}_candidates.json", "w", encoding="utf-8"
    ) as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print(
        f"{image_path.name}: slices={slice_count} raw={len(raw_candidates)} "
        f"merged={len(candidates)} saved_slices={len(slice_records)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="1280-slice fabric anomaly candidate prefilter"
    )
    parser.add_argument("--input", required=True, help="input image or image directory")
    parser.add_argument("--output", default="output", help="output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()

    images = list_images(input_path)
    if not images:
        raise SystemExit(f"no images found: {input_path}")

    for image_path in images:
        process_image(image_path, output_dir)


if __name__ == "__main__":
    main()
