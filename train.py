import logging
import os
from pathlib import Path
from time import time

import torch
from anomalib.callbacks import ModelCheckpoint
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Stfpm

from Trash.utils import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


class StfpmModelExporter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    @staticmethod
    def _get_output(outputs: object, *names: str) -> torch.Tensor:
        for name in names:
            if isinstance(outputs, dict) and name in outputs:
                return outputs[name]
            if hasattr(outputs, name):
                return getattr(outputs, name)
        available = list(outputs.keys()) if isinstance(outputs, dict) else dir(outputs)
        raise RuntimeError(f"STFPM 输出中缺少字段 {names}, available={available}")

    def forward(
        self, input_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(input_tensor)
        score = self._get_output(outputs, "pred_score", "output")
        anomaly_map = self._get_output(outputs, "anomaly_map")
        pred_label = self._get_output(outputs, "pred_label")
        pred_mask = self._get_output(outputs, "pred_mask")
        return score, anomaly_map, pred_label, pred_mask


def simplify_onnx(onnx_path: Path, image_size: int) -> None:
    try:
        import onnx
        from onnxsim import simplify
    except ImportError as exc:
        raise RuntimeError("ONNX 简化需要安装 onnx 和 onnxsim。") from exc

    onnx_model = onnx.load(str(onnx_path))
    simplified_model, ok = simplify(
        onnx_model,
        overwrite_input_shapes={"input": [1, 3, image_size, image_size]},
    )
    if not ok:
        raise RuntimeError("onnx simplify failed")
    onnx.save(simplified_model, str(onnx_path))


def train(train_root):
    datamodule = Folder(
        name="RDK captured",
        root=train_root,
        normal_dir="good",
        num_workers=3,
    )
    max_epochs = 3

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
    image_size = 256
    dummy = torch.rand(1, 3, image_size, image_size)
    exporter = StfpmModelExporter(model).eval().cpu()
    output_names = ["output", "anomaly_map", "483", "485"]

    torch.onnx.export(
        exporter,
        dummy,
        str(onnx_path),
        opset_version=11,
        input_names=["input"],
        output_names=output_names,
    )
    simplify_onnx(onnx_path, image_size)
    logger.info("转换成 STFPM onnx 并简化成功: %s", onnx_path)

    return onnx_path
