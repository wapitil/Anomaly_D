from __future__ import annotations

import argparse

import numpy as np
import torch
from anomalib.models import Stfpm

from predict_image_utils import (
    build_visualization,
    ensure_file_exists,
    load_image,
    preprocess_for_model,
    print_summary,
    save_or_show,
)

DEFAULT_CKPT_PATH = (
    "/home/wapiti/Projects/Anomaly_D/results/Stfpm/MVTecAD/leather/latest/weights/"
    "lightning/model.ckpt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 STFPM ckpt 模型做单图验证。")
    parser.add_argument("image_path", help="输入图片路径")
    parser.add_argument("output_path", nargs="?", default=None, help="可选输出图片路径")
    parser.add_argument("--model-path", default=DEFAULT_CKPT_PATH, help="ckpt 模型路径")
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


def predict_ckpt(model_path: str, input_nchw: np.ndarray) -> tuple[float, np.ndarray]:
    ensure_file_exists(model_path, "ckpt模型")
    model = Stfpm.load_from_checkpoint(
        model_path,
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
    )
    model.eval().cpu()

    with torch.no_grad():
        outputs = model(torch.from_numpy(input_nchw).float())

    score = None
    anomaly_map = None

    if isinstance(outputs, dict):
        if "pred_score" in outputs:
            score = float(outputs["pred_score"].detach().cpu().numpy().reshape(-1)[0])
        elif "output" in outputs:
            score = float(outputs["output"].detach().cpu().numpy().reshape(-1)[0])
        if "anomaly_map" in outputs:
            anomaly_map = outputs["anomaly_map"].detach().cpu().numpy().astype(np.float32)
    else:
        if hasattr(outputs, "pred_score"):
            score = float(outputs.pred_score.detach().cpu().numpy().reshape(-1)[0])
        elif hasattr(outputs, "output"):
            score = float(outputs.output.detach().cpu().numpy().reshape(-1)[0])
        if hasattr(outputs, "anomaly_map"):
            anomaly_map = outputs.anomaly_map.detach().cpu().numpy().astype(np.float32)

    if score is None or anomaly_map is None:
        raise RuntimeError("CKPT 输出中未找到 score 或 anomaly_map，请检查 anomalib 版本。")

    return score, anomaly_map


def main() -> None:
    args = parse_args()
    image = load_image(args.image_path)
    input_nchw = preprocess_for_model(image, args.image_size)
    score, anomaly_map = predict_ckpt(args.model_path, input_nchw)

    combined, _, big_defects, small_defects = build_visualization(
        image,
        score,
        anomaly_map,
        args.score_thresh,
        args.big_area_thresh,
        args.min_area,
        args.pad,
    )
    print_summary("ckpt", score, big_defects, small_defects)
    save_or_show(args.output_path, combined)


if __name__ == "__main__":
    main()
