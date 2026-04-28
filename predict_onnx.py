from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def split_defects_by_mask(
    img: np.ndarray,
    mask: np.ndarray,
    big_area_thresh: int = 200,
    min_area: int = 20,
    pad: int = 10,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """
    根据 mask 把异常区域分成“大缺陷”和“小缺陷”。
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    del labels, centroids

    big_defects: list[dict[str, object]] = []
    small_defects: list[dict[str, object]] = []

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

        roi = img[y1:y2, x1:x2]
        info: dict[str, object] = {
            "area": area,
            "bbox": (x1, y1, x2, y2),
            "roi": roi,
        }

        if area >= big_area_thresh:
            big_defects.append(info)
        else:
            small_defects.append(info)

    return big_defects, small_defects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RDK X5 适配版 STFPM 单图推理脚本，默认使用 ONNX Runtime CPU。"
    )
    parser.add_argument(
        "--image_path", default="./leather/glue/010.png"
    )
    parser.add_argument(
        "--output_path",
        nargs="?",
        default="./onnx_rdkx5_result.png",
        help="可选输出图片路径，不传则自动生成 *_rdkx5_result.png",
    )
    parser.add_argument(
        "--model-path",
        default="stfpm_leather_split.onnx",
        help="ONNX 模型路径，默认 stfpm_leather.onnx",
    )
    parser.add_argument(
        "--backend",
        choices=("onnx", "ckpt"),
        default="onnx",
        help="推理后端，RDK X5 上建议使用 onnx",
    )
    parser.add_argument("--image-size", type=int, default=256, help="模型输入尺寸")
    parser.add_argument(
        "--big-area-thresh", type=int, default=2000, help="大缺陷面积阈值"
    )
    parser.add_argument(
        "--min-area", type=int, default=200, help="忽略的小噪点面积阈值"
    )
    parser.add_argument("--pad", type=int, default=10, help="裁剪 ROI 向外扩展像素")
    parser.add_argument(
        "--show",
        action="store_true",
        help="显示窗口。RDK X5 无桌面环境时不建议开启",
    )
    return parser.parse_args()


def preprocess_for_model(
    image_bgr: np.ndarray, image_size: int
) -> tuple[np.ndarray, np.ndarray]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size))
    input_float = resized.astype(np.float32) / 255.0
    input_chw = np.transpose(input_float, (2, 0, 1))
    input_nchw = np.expand_dims(input_chw, axis=0)
    return resized, input_nchw


def build_onnx_session(model_path: str):
    import onnxruntime as ort

    # RDK X5 常见部署是板端 CPU 推理，这里显式指定 CPU provider，
    # 并抬高日志级别以屏蔽 GPU 设备探测告警。
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    return ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


def run_onnx(model_path: str, input_nchw: np.ndarray) -> dict[str, np.ndarray | float | bool]:
    session = build_onnx_session(model_path)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    output_values = session.run(output_names, {input_name: input_nchw})
    outputs = dict(zip(output_names, output_values))

    score = float(np.asarray(outputs["output"]).reshape(-1)[0])
    anomaly_map = np.asarray(outputs["anomaly_map"], dtype=np.float32)
    pred_label = bool(np.asarray(outputs.get("483", [[score >= 0.5]])).reshape(-1)[0])
    model_mask = np.asarray(outputs.get("485", np.zeros_like(anomaly_map)), dtype=bool)

    if score is None or anomaly_map is None:
        raise RuntimeError("ONNX 输出中未找到 output 或 anomaly_map。")

    return {
        "score": score,
        "anomaly_map": anomaly_map,
        "pred_label": pred_label,
        "model_mask": model_mask,
    }


def run_ckpt(model_path: str, input_nchw: np.ndarray) -> dict[str, np.ndarray | float]:
    import torch
    from anomalib.models import Stfpm

    model = Stfpm.load_from_checkpoint(
        model_path,
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
    )
    model.eval().cpu()

    input_tensor = torch.from_numpy(input_nchw).float()
    with torch.no_grad():
        outputs = model(input_tensor)

    score = None
    anomaly_map = None

    if isinstance(outputs, dict):
        if "pred_score" in outputs:
            score = float(outputs["pred_score"].detach().cpu().numpy().reshape(-1)[0])
        elif "output" in outputs:
            score = float(outputs["output"].detach().cpu().numpy().reshape(-1)[0])

        if "anomaly_map" in outputs:
            anomaly_map = (
                outputs["anomaly_map"].detach().cpu().numpy().astype(np.float32)
            )
    else:
        if hasattr(outputs, "pred_score"):
            score = float(outputs.pred_score.detach().cpu().numpy().reshape(-1)[0])
        elif hasattr(outputs, "output"):
            score = float(outputs.output.detach().cpu().numpy().reshape(-1)[0])

        if hasattr(outputs, "anomaly_map"):
            anomaly_map = outputs.anomaly_map.detach().cpu().numpy().astype(np.float32)

    if score is None or anomaly_map is None:
        raise RuntimeError(
            "CKPT 输出中未找到 score 或 anomaly_map，请检查 anomalib 版本。"
        )

    return {
        "score": score,
        "anomaly_map": anomaly_map,
    }


def make_results_dict(
    image_bgr: np.ndarray,
    score: float,
    anomaly_map: np.ndarray,
    pred_label: bool,
    model_mask: np.ndarray,
) -> dict[str, np.ndarray | float]:
    map_2d = np.squeeze(anomaly_map).astype(np.float32)
    map_min = float(map_2d.min())
    map_max = float(map_2d.max())
    map_mean = float(map_2d.mean())
    map_std = float(map_2d.std())

    if map_max > map_min:
        map_norm = (map_2d - map_min) / (map_max - map_min)
    else:
        map_norm = np.zeros_like(map_2d, dtype=np.float32)

    if pred_label:
        mask_small = np.squeeze(model_mask).astype(np.uint8) * 255
    else:
        mask_small = np.zeros_like(map_2d, dtype=np.uint8)

    heatmap_small = cv2.applyColorMap(
        (map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap = cv2.resize(heatmap_small, (image_bgr.shape[1], image_bgr.shape[0]))
    mask = cv2.resize(
        mask_small,
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    vis = cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)
    cv2.putText(
        vis,
        f"score={score:.4f}, pred={pred_label}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    return {
        "score": score,
        "anomaly_map": map_2d,
        "mask": mask,
        "vis": vis,
        "map_min": map_min,
        "map_max": map_max,
        "map_mean": map_mean,
        "map_std": map_std,
        "pred_label": pred_label,
        "model_mask_px": int(np.count_nonzero(mask_small)),
    }


def resolve_output_path(image_path: str, output_path: str | None) -> str:
    if output_path:
        return output_path

    image_file = Path(image_path)
    return str(image_file.with_name(f"{image_file.stem}_rdkx5_result.png"))


def can_show_window() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def main():
    args = parse_args()

    image_path = args.image_path
    model_path = args.model_path
    output_path = resolve_output_path(image_path, args.output_path)

    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
        return
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    _, input_nchw = preprocess_for_model(img, args.image_size)

    if args.backend != "onnx":
        print("模型自带 mask 只支持 ONNX 后端，请使用 --backend onnx")
        return

    backend_outputs = run_onnx(model_path, input_nchw)

    results = make_results_dict(
        img,
        float(backend_outputs["score"]),
        np.asarray(backend_outputs["anomaly_map"], dtype=np.float32),
        bool(backend_outputs["pred_label"]),
        np.asarray(backend_outputs["model_mask"], dtype=bool),
    )
    vis = results["vis"]

    mask = results["mask"]
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    big_defects, small_defects = split_defects_by_mask(
        img,
        mask,
        big_area_thresh=args.big_area_thresh,
        min_area=args.min_area,
        pad=args.pad,
    )

    print("===================================")
    print(f"后端 backend: {args.backend}")
    print(f"模型路径: {model_path}")
    print(f"图像分数 score: {results['score']:.4f}")
    print(f"模型图像级判断 pred_label: {results['pred_label']}")
    print(
        "异常图统计: "
        f"min={results['map_min']:.6f}, "
        f"max={results['map_max']:.6f}, "
        f"mean={results['map_mean']:.6f}, "
        f"std={results['map_std']:.6f}"
    )
    print(f"模型自带 mask 像素数: {results['model_mask_px']}")
    print(f"大缺陷数量: {len(big_defects)}")
    print(f"小缺陷数量: {len(small_defects)}")

    for i, d in enumerate(big_defects):
        print(f"[大缺陷 {i}] area={d['area']}, bbox={d['bbox']}")

    for i, d in enumerate(small_defects):
        print(f"[小缺陷 {i}] area={d['area']}, bbox={d['bbox']}")
    print("===================================")

    dispatch_vis = img.copy()

    for d in big_defects:
        x1, y1, x2, y2 = d["bbox"]
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

    for d in small_defects:
        x1, y1, x2, y2 = d["bbox"]
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

    h, w = img.shape[:2]
    mask_3ch = cv2.resize(mask_3ch, (w, h), interpolation=cv2.INTER_NEAREST)
    vis = cv2.resize(vis, (w, h))
    dispatch_vis = cv2.resize(dispatch_vis, (w, h))

    combined = np.hstack([img, mask_3ch, vis, dispatch_vis])

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

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(output_path, combined)
    print(f"结果已保存到: {output_path}")

    if args.show:
        if can_show_window():
            cv2.imshow("Original | Mask | Result | Dispatch", combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("未检测到桌面显示环境，跳过窗口显示。")


if __name__ == "__main__":
    main()
