from time import time

import colorlog
from anomalib.data import Folder, MVTecAD
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
logger.setLevel("INFO")


def train(is_real, receiver_root):
    # 1. 数据
    if not is_real:
        datamodule = MVTecAD(
            root=receiver_root,
            category="leather",
            train_batch_size=64,
            eval_batch_size=64,
            num_workers=2,
        )
    else:
        datamodule = Folder(
            name="RDK captured",
            root=receiver_root,
            normal_dir="good",
            num_workers=2,
        )
        max_epochs = 3

    # 2. 模型（重点）
    model = Stfpm(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
    # model = Fastflow(backbone="resnet18", pre_trained=True, flow_steps=8)

    # 3. 训练

    engine = Engine(
        max_epochs=max_epochs,
    )

    start_time = time()
    engine.fit(model=model, datamodule=datamodule)
    end_time = time()
    train_time = end_time - start_time

    logger.info("Train Time: %.2f seconds", train_time)
