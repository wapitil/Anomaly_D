from pathlib import Path

import torch
from anomalib.models import Stfpm


CKPT_PATH = "results/Stfpm/MVTecAD/leather/v0/weights/lightning/model.ckpt"
OUTPUT_DIR = "results/Stfpm/MVTecAD/leather/v0/weights/lightning"
FEATURE_ONNX_NAME = "stfpm_features.onnx"

IMAGE_SIZE = 256
BACKBONE = "resnet18"
LAYERS = ["layer1", "layer2", "layer3"]
OPSET_VERSION = 11
SIMPLIFY = True


class StfpmFeatureExporter(torch.nn.Module):
    def __init__(self, stfpm_model: torch.nn.Module, layers: list[str]) -> None:
        super().__init__()
        self.stfpm_model = stfpm_model
        self.layers = layers

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        teacher_features = self.stfpm_model.teacher_model(input_tensor)
        student_features = self.stfpm_model.student_model(input_tensor)
        outputs: list[torch.Tensor] = []
        for layer in self.layers:
            outputs.append(teacher_features[layer])
            outputs.append(student_features[layer])
        return tuple(outputs)


def simplify_onnx(onnx_path: Path, image_size: int) -> None:
    try:
        import onnx
        from onnxsim import simplify
    except ImportError as exc:
        raise RuntimeError("SIMPLIFY=True 需要安装 onnx 和 onnxsim。") from exc

    onnx_model = onnx.load(str(onnx_path))
    simplified_model, ok = simplify(
        onnx_model,
        overwrite_input_shapes={"input": [1, 3, image_size, image_size]},
    )
    if not ok:
        raise RuntimeError("onnx simplify failed")
    onnx.save(simplified_model, str(onnx_path))


def main() -> None:
    ckpt_path = Path(CKPT_PATH)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = Stfpm.load_from_checkpoint(
            str(ckpt_path),
            backbone=BACKBONE,
            layers=LAYERS,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "model.gaussian" in message or "model.feature_extractor" in message:
            raise RuntimeError(
                f"当前 CKPT_PATH 看起来不是 STFPM checkpoint: {ckpt_path}. "
                "请改成 Stfpm 训练得到的 ckpt，不能使用 PaDiM ckpt。"
            ) from exc
        raise
    model.eval().cpu()

    stfpm_model = model.model
    stfpm_model.eval().cpu()

    exporter = StfpmFeatureExporter(stfpm_model, LAYERS).eval().cpu()
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    output_names = [
        f"{prefix}_{layer}"
        for layer in LAYERS
        for prefix in ("teacher", "student")
    ]

    onnx_path = output_dir / FEATURE_ONNX_NAME
    torch.onnx.export(
        exporter,
        dummy,
        str(onnx_path),
        opset_version=OPSET_VERSION,
        input_names=["input"],
        output_names=output_names,
    )

    if SIMPLIFY:
        simplify_onnx(onnx_path, IMAGE_SIZE)

    with torch.no_grad():
        feature_shapes = [tuple(int(v) for v in tensor.shape) for tensor in exporter(dummy)]

    print(f"Exported STFPM feature ONNX: {onnx_path}")
    for name, shape in zip(output_names, feature_shapes, strict=True):
        print(f"{name}: shape={shape}")


if __name__ == "__main__":
    main()
