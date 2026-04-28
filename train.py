import logging
import os
from pathlib import Path
from time import time

import torch
from anomalib.callbacks import ModelCheckpoint
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Stfpm


from utils import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


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


def train(train_root):
    datamodule = Folder(
        name="RDK captured",
        root=train_root,
        normal_dir="good",
        num_workers=3,
    )
    max_epochs = 50

    # 2. 模型（重点）
    layers = ["layer1", "layer2", "layer3"]
    model = Stfpm(backbone="resnet18", layers=layers)
    # model = Fastflow(backbone="resnet18", pre_trained=True, flow_steps=8)

    # 3. 训练
    model_path = Path(train_root) / "models"
    os.makedirs(model_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path,  # 指定模型保存的路径
        filename="model",
        save_last=False,
    )

    engine = Engine(
        max_epochs=max_epochs, callbacks=[checkpoint_callback], logger=False
    )

    start = time()

    # 开始训练

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    end = time()
    train_time = end - start

    logger.info("Train Time: %.2f seconds", train_time)
    logger.info("模型训练完成，正在将模型转换至 onnx ")

    onnx_path = Path(model_path) / "stfpm_RDK.onnx"

    model.eval().cpu()
    dummy = torch.randn(1, 3, 256, 256)
    exporter = StfpmFeatureExporter(model.model, layers).eval().cpu()
    output_names = [
        f"{prefix}_{layer}" for layer in layers for prefix in ("teacher", "student")
    ]

    torch.onnx.export(
        exporter,
        dummy,
        str(onnx_path),
        opset_version=11,
        input_names=["input"],
        output_names=output_names,
    )
    logger.info("转换成 STFPM feature onnx 成功: %s", onnx_path)

    return onnx_path
