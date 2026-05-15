from __future__ import annotations

from pathlib import Path

import torch
from core import get_backbone_config, load_feature_model

# PC: export only the backbone feature extractor to ONNX.
# Output: runs/<PROJECT>/onnx/<PROJECT>.onnx for OpenExplorer compile.

BACKBONE_NAME = "resnet18"
PROJECT = "res_640"
ONNX_PATH = Path("runs") / PROJECT / "onnx" / f"{PROJECT}.onnx"
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
        external_data=False,
        dynamo=False,
    )

    print(f"exported: {ONNX_PATH}")
    print("BPU model output is a feature vector, not a final anomaly label.")


if __name__ == "__main__":
    main()
