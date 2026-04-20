import logging
from pathlib import Path
from time import time

import colorlog
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Stfpm

# from anomalib.models import Fastflow

# 配置彩色日志
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
)

logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

dataset_root = Path.cwd() / "Datasets" / "MVTecAD"

# 1. 数据
datamodule = MVTecAD(
    root=dataset_root,
    category="leather",
    train_batch_size=64,
    eval_batch_size=64,
    num_workers=2,
)


# 2. 模型（重点）
model = Stfpm(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
# model = Fastflow(backbone="resnet18", pre_trained=True, flow_steps=8)

# 3. 训练
engine = Engine(
    max_epochs=50,
)

start_time = time()
engine.fit(model=model, datamodule=datamodule)
end_time = time()
train_time = end_time - start_time
logging.info(f"Train Time :{train_time}")

# 4. 推理
# predictions = engine.predict(model=model, datamodule=datamodule)
