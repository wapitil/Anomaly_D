import torch
from anomalib.models import Stfpm

ckpt_path = "results/Stfpm/MVTecAD/bottle/latest/weights/lightning/model.ckpt"

model = Stfpm.load_from_checkpoint(
    ckpt_path,
    backbone="resnet18",
    layers=["layer1", "layer2", "layer3"],
)
model.eval().cpu()

dummy = torch.randn(1, 3, 256, 256)

torch.onnx.export(
    model,
    dummy,
    "stfpm_static.onnx",
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
)