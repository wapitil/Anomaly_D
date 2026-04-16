from anomalib.models import Stfpm
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from pathlib import Path

dataset_root = Path.cwd() / "Datasets" / "MVTecAD"
# 1. 数据
datamodule = MVTecAD(
    root=dataset_root,
    category="bottle",
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=0,
)


# 2. 模型（重点）
model = Stfpm(
    backbone="resnet18",
    layers=["layer1", "layer2", "layer3"]
)

# 3. 训练
engine = Engine(max_epochs=12)
engine.fit(model=model, datamodule=datamodule)

# 4. 推理
predictions = engine.predict(model=model, datamodule=datamodule)