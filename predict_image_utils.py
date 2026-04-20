from __future__ import annotations

import os

import cv2
import numpy as np


def ensure_file_exists(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label}不存在: {path}")


def load_image(image_path: str) -> np.ndarray:
    ensure_file_exists(image_path, "图片")
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"无法读取图片: {image_path}")
    return image


def preprocess_for_model(image_bgr: np.ndarray, image_size: int) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size))
    input_float = resized.astype(np.float32) / 255.0
    input_chw = np.transpose(input_float, (2, 0, 1))
    return np.expand_dims(input_chw, axis=0)


def split_defects_by_mask(
    img: np.ndarray,
    mask: np.ndarray,
    big_area_thresh: int = 200,
    min_area: int = 20,
    pad: int = 10,
) -> tuple[list[dict], list[dict]]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    big_defects = []
    small_defects = []
    h, w = img.shape[:2]

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + ww + pad)
        y2 = min(h, y + hh + pad)

        info = {
            "area": area,
            "bbox": (x1, y1, x2, y2),
            "roi": img[y1:y2, x1:x2],
        }
        if area >= big_area_thresh:
            big_defects.append(info)
        else:
            small_defects.append(info)

    return big_defects, small_defects


def normalize_anomaly_map(anomaly_map: np.ndarray) -> np.ndarray:
    map_2d = np.squeeze(anomaly_map).astype(np.float32)
    map_min = float(map_2d.min())
    map_max = float(map_2d.max())
    if map_max <= map_min:
        return np.zeros_like(map_2d, dtype=np.float32)
    return (map_2d - map_min) / (map_max - map_min)


def build_visualization(
    image_bgr: np.ndarray,
    score: float,
    anomaly_map: np.ndarray,
    score_thresh: float,
    big_area_thresh: int,
    min_area: int,
    pad: int,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    map_norm = normalize_anomaly_map(anomaly_map)
    mask_small = (map_norm >= score_thresh).astype(np.uint8) * 255
    heatmap_small = cv2.applyColorMap(
        (map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
    )

    h, w = image_bgr.shape[:2]
    heatmap = cv2.resize(heatmap_small, (w, h))
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    vis = cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)
    cv2.putText(
        vis,
        f"score={score:.4f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    big_defects, small_defects = split_defects_by_mask(
        image_bgr,
        mask,
        big_area_thresh=big_area_thresh,
        min_area=min_area,
        pad=pad,
    )

    dispatch_vis = image_bgr.copy()
    for defect in big_defects:
        x1, y1, x2, y2 = defect["bbox"]
        cv2.rectangle(dispatch_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            dispatch_vis,
            "BIG",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    for defect in small_defects:
        x1, y1, x2, y2 = defect["bbox"]
        cv2.rectangle(dispatch_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            dispatch_vis,
            "SMALL",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([image_bgr, mask_3ch, vis, dispatch_vis])

    cv2.putText(
        combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    cv2.putText(
        combined, "Mask", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    cv2.putText(
        combined,
        "Result",
        (2 * w + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        combined,
        "Dispatch",
        (3 * w + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    return combined, mask, big_defects, small_defects


def save_or_show(output_path: str | None, combined: np.ndarray) -> None:
    if output_path:
        cv2.imwrite(output_path, combined)
        print(f"结果已保存到: {output_path}")
        return

    cv2.imshow("Original | Mask | Result | Dispatch", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_summary(
    backend: str,
    score: float,
    big_defects: list[dict],
    small_defects: list[dict],
) -> None:
    print("===================================")
    print(f"后端 backend: {backend}")
    print(f"图像分数 score: {score:.4f}")
    print(f"大缺陷数量: {len(big_defects)}")
    print(f"小缺陷数量: {len(small_defects)}")

    for i, defect in enumerate(big_defects):
        print(f"[大缺陷 {i}] area={defect['area']}, bbox={defect['bbox']}")

    for i, defect in enumerate(small_defects):
        print(f"[小缺陷 {i}] area={defect['area']}, bbox={defect['bbox']}")

    print("===================================")
