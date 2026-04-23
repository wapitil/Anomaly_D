from pathlib import Path

import onnx
import torch
from anomalib.models import Stfpm
from onnxsim import simplify

CKPT_PATH = Path("results/Stfpm/MVTecAD/leather/latest/weights/lightning/model.ckpt")
ONNX_PATH = Path("stfpm_leather.onnx")
IMAGE_SIZE = 256
OPSET = 11

model = Stfpm.load_from_checkpoint(
    str(CKPT_PATH),
    backbone="resnet18",
    layers=["layer1", "layer2", "layer3"],
)
model.eval().cpu()

dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

torch.onnx.export(
    model,
    dummy,
    str(ONNX_PATH),
    opset_version=OPSET,
    input_names=["input"],
    output_names=["output"],
)

onnx_model = onnx.load(str(ONNX_PATH))
simplified_model, ok = simplify(
    onnx_model,
    overwrite_input_shapes={"input": [1, 3, IMAGE_SIZE, IMAGE_SIZE]},
)

if not ok:
    raise RuntimeError("onnx simplify failed")

onnx.save(simplified_model, str(ONNX_PATH))

print(f"done: {ONNX_PATH}")
