from __future__ import annotations

import argparse

import numpy as np
import onnxruntime
from predict_image_utils import (
    build_visualization,
    ensure_file_exists,
    load_image,
    preprocess_for_model,
    print_summary,
    save_or_show,
)

DEFAULT_ONNX_PATH = "stfpm_leather_split.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 STFPM onnx 模型做单图验证。")
    parser.add_argument("image_path", help="输入图片路径")
    parser.add_argument("output_path", nargs="?", default=None, help="可选输出图片路径")
    parser.add_argument("--model-path", default=DEFAULT_ONNX_PATH, help="onnx 模型路径")
    parser.add_argument("--image-size", type=int, default=256, help="模型输入尺寸")
    parser.add_argument(
        "--score-thresh", type=float, default=0.5, help="mask 二值化阈值，范围 0~1"
    )
    parser.add_argument(
        "--big-area-thresh", type=int, default=2000, help="大缺陷面积阈值"
    )
    parser.add_argument("--min-area", type=int, default=20, help="忽略的小噪点面积阈值")
    parser.add_argument("--pad", type=int, default=10, help="裁剪 ROI 向外扩展像素")
    return parser.parse_args()


def predict_onnx(model_path: str, input_nchw: np.ndarray) -> tuple[float, np.ndarray]:
    ensure_file_exists(model_path, "onnx模型")
    session = onnxruntime.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_nchw})

    score = None
    anomaly_map = None
    for output_meta, output_value in zip(session.get_outputs(), outputs):
        if output_meta.name == "output":
            score = float(np.asarray(output_value).reshape(-1)[0])
        elif output_meta.name == "anomaly_map":
            anomaly_map = np.asarray(output_value, dtype=np.float32)

    if score is None or anomaly_map is None:
        raise RuntimeError("ONNX 输出中未找到 output 或 anomaly_map。")

    return score, anomaly_map


def main() -> None:
    args = parse_args()
    image = load_image(args.image_path)
    input_nchw = preprocess_for_model(image, args.image_size)
    score, anomaly_map = predict_onnx(args.model_path, input_nchw)

    combined, _, big_defects, small_defects = build_visualization(
        image,
        score,
        anomaly_map,
        args.score_thresh,
        args.big_area_thresh,
        args.min_area,
        args.pad,
    )
    print_summary("onnx", score, big_defects, small_defects)
    save_or_show(args.output_path, combined)


if __name__ == "__main__":
    main()
