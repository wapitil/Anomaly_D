from __future__ import annotations

import logging
from pathlib import Path

import torch

logging.getLogger("torch").setLevel(logging.ERROR)


def maybe_simplify(onnx_path: Path, image_size: int) -> None:
    try:
        import onnx
        from onnxsim import simplify
    except ImportError as exc:
        raise RuntimeError(
            "Requested --simplify, but onnx-simplifier is not installed."
        ) from exc

    model = onnx.load(str(onnx_path))
    simplified, check = simplify(
        model,
        overwrite_input_shapes={"input": [1, 3, image_size, image_size]},
    )
    if not check:
        raise RuntimeError("onnx-simplifier reported validation failure.")
    onnx.save(simplified, str(onnx_path))


def export_model_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    image_size: int,
    opset: int,
    simplify: bool,
) -> None:
    model.eval().cpu()
    dummy = torch.randn(1, 3, image_size, image_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
    )

    if simplify:
        maybe_simplify(output_path, image_size)

    print(f"Exported ONNX to {output_path}")
    if simplify:
        print("Applied onnx-simplifier.")
