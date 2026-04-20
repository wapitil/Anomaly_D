from __future__ import annotations

import argparse
from pathlib import Path

from anomalib.models import Fastflow

from export_utils import export_model_to_onnx

DEFAULT_CKPT = Path(
    "results/Fastflow/MVTecAD/leather/latest/weights/lightning/model.ckpt"
)
DEFAULT_OUTPUT = Path("fastflow_leather.onnx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FastFlow checkpoint to ONNX.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=DEFAULT_CKPT,
        help="Path to the FastFlow checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square input image size.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Run onnx-simplifier after export if available.",
    )
    return parser.parse_args()


def build_model(ckpt_path: Path) -> Fastflow:
    return Fastflow.load_from_checkpoint(
        str(ckpt_path),
        backbone="resnet18",
        pre_trained=True,
        flow_steps=8,
    )


def main() -> None:
    args = parse_args()
    model = build_model(args.ckpt)
    export_model_to_onnx(
        model=model,
        output_path=args.output,
        image_size=args.image_size,
        opset=args.opset,
        simplify=args.simplify,
    )


if __name__ == "__main__":
    main()
