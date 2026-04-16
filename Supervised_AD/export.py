""" 
Load: 
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, weights_only=True))
model.eval()
参考链接：
https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
"""
import torch 
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from pathlib import Path
from onnxsim import simplify
import onnx
# Watch: 硬编码绝对路径，换机器需手动修改
OUT_DIR = Path("/home/wapiti/Projects/Anomaly_D/vis_outputs")
# Why: 224 是 MobileNetV3 官方推荐输入尺寸，变更会导致精度下降
MODEL_INPUT_SIZE = 224
device = "cuda"


# Why: torch.load 只恢复权重张量，不保存模型结构，所以必须先手动重建完全一致的架构
#      架构不匹配会导致 load_state_dict 报错或静默产生错误权重
model = mobilenet_v3_small()
# Watch: 分类头必须与 train_gate_rdk.py 中训练时的修改完全一致（1000→1），
#        否则权重形状不匹配会直接报错
last_layer = model.classifier[3]
if isinstance(last_layer, nn.Linear):
    model.classifier[3] = nn.Linear(last_layer.in_features, 1)

model.load_state_dict(torch.load("../models/best_gate_model.pth", weights_only=True))
# Why: 切换到 eval 模式以关闭 Dropout 和 BatchNorm 的训练行为，确保导出结果确定性
model.eval()

dummy_input = torch.randn(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, device=device)

torch.onnx.export(
    model,
    dummy_input,
    str(OUT_DIR / "gate_tinynet_4x4.onnx"),
    input_names=["images"],
    output_names=["scores"],
    # Why: opset 11 对 hardsigmoid/hardswish 等 MobileNetV3 轻量算子支持完善，
    #      低于 11 可能导致算子回退到复杂实现，影响 BPU 兼容性
    opset_version=11,
    # Why: 常量折叠将推理时不变的子图预计算，减小 ONNX 文件体积并加速推理
    do_constant_folding=True,
)

# 4. 简化 ONNX（RDK 工具链要求）
model_simp, check = simplify("gate_mobilenetv3.onnx")
if check:
    onnx.save(model_simp, "gate_mobilenetv3_sim.onnx")