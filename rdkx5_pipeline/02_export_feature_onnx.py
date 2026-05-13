from __future__ import annotations

from pathlib import Path

import torch
from core import get_backbone_config, load_feature_model, save_metadata

# This exports only the backbone feature extractor.
# OpenExplorer will quantize and compile this ONNX into an RDK X5 BPU model.
BACKBONE_NAME = "resnet18"
PROJECT = "res_640"
ONNX_PATH = Path("runs") / PROJECT / f"{PROJECT}.onnx"
OPSET_VERSION = 11


def main() -> None:
    config = get_backbone_config(BACKBONE_NAME)
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = load_feature_model(BACKBONE_NAME).cpu()
    dummy_input = torch.randn(1, 3, config.input_size, config.input_size)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["feature"],
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
    )
    save_metadata(
        ONNX_PATH.with_suffix(".metadata.json"),
        BACKBONE_NAME,
        {
            "onnx_file": ONNX_PATH.name,
            "input_layout": "NCHW",
            "input_color": "RGB",
            "input_dtype": "float32",
            "preprocess": "letterbox_resize_to_square_then_imagenet_mean_std",
            "output": "raw_feature_before_l2_normalize",
            "next_step": "Use OpenExplorer to quantize and compile for RDK X5.",
        },
    )

    print(f"exported: {ONNX_PATH}")
    print("BPU model output is a feature vector, not a final anomaly label.")


if __name__ == "__main__":
    main()
