# -*- coding: utf-8 -*-
"""
ONNX 模型导出与简化工具
========================
本脚本的功能：
  阶段1: 创建模型并导出为 ONNX 格式
  阶段2: 使用 onnxsim 简化 ONNX 模型（去除冗余算子）
  阶段3: 统计 ONNX 模型的参数量

注意：本脚本目前导出的是 timm 预训练的 MobileNetV3-Small（1000类），
      而非我们训练好的 Gate 二分类模型。
      如需导出训练好的 Gate 模型，见文件末尾说明。
"""

import torch
import torch.onnx
import onnx
from onnxsim import simplify
from timm.models import create_model
from timm.models.mobilenetv3 import mobilenet_v3_small


def count_parameters(onnx_model_path):
    """
    阶段3: 统计 ONNX 模型的总参数量
    ─────────────────────────────────
    原理：遍历 ONNX 模型图中所有 initializer（即权重张量），
         计算每个张量的元素数量并累加。

    参数:
        onnx_model_path: ONNX 模型文件路径

    返回:
        total_params: 模型总参数量（标量）
    """
    # 加载 ONNX 模型（解析 protobuf 格式）
    model = onnx.load(onnx_model_path)

    # model.graph.initializer 包含模型所有权重（Conv权重、BN参数、FC权重等）
    initializer = model.graph.initializer

    total_params = 0
    for tensor in initializer:
        # tensor.dims 是权重张量的形状，如 [64, 3, 3, 3] 表示 3×3 卷积，64个输出通道
        dims = tensor.dims
        # 计算该权重张量的元素总数 = 各维度之积
        params = 1
        for dim in dims:
            params *= dim
        total_params += params

    return total_params


if __name__ == "__main__":

    # ================================================================
    # 阶段1: 创建模型 → 导出 ONNX
    # ================================================================
    # 检测是否有 GPU 可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ⚠️ 这里创建的是 timm 预训练的 MobileNetV3-Small（1000类 ImageNet 分类器）
    # 并不是我们训练好的 Gate 二分类模型！
    model = create_model('mobilenet_v3_small', pretrained=True)
    model.eval()  # 切换到评估模式（关闭 Dropout、BN 使用运行均值）

    # 构造一个假的输入张量，形状为 (1, 3, 224, 224)
    # 1 = batch_size, 3 = RGB通道, 224×224 = 输入分辨率
    # 注意：device 设为 "cpu"，因为 ONNX 导出在 CPU 上更稳定
    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")

    # 导出后的 ONNX 文件路径
    onnx_file_path = "mobilenet_v3_small.onnx"

    # ── torch.onnx.export 核心参数解释 ──
    # model       : 要导出的 PyTorch 模型
    # dummy_input : 模型的示例输入（用于追踪计算图，不会参与训练）
    # onnx_file_path : 输出文件路径
    # opset_version  : ONNX 算子集版本（11 兼容性较好）
    # verbose=True   : 打印计算图的详细信息（调试用）
    # input_names    : 给输入节点起名（后续推理时通过名字访问）
    # output_names   : 给输出节点起名
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        opset_version=11,
        verbose=True,
        input_names=["data"],      # 输入名: data
        output_names=["output"],   # 输出名: output
    )

    # ================================================================
    # 阶段2: 使用 onnxsim 简化 ONNX 模型
    # ================================================================
    # onnxsim 会做以下优化：
    #   - 常量折叠（Constant Folding）：将编译期可确定的计算提前算好
    #   - 算子融合（Operator Fusion）：如 Conv+BN 融合为单个 Conv
    #   - 死代码消除：移除没有输出的冗余节点
    # 简化后的模型更小、推理更快，且对 RDK 工具链更友好
    model_simp, check = simplify(onnx_file_path)

    if check:
        print("Simplified model is valid.")
        # ⚠️ 文件名有误：导出的是 small 但保存为 large
        simplified_onnx_file_path = "mobilenetv3_large_100.onnx"
        onnx.save(model_simp, simplified_onnx_file_path)
        print(f"Simplified model saved to {simplified_onnx_file_path}")
    else:
        print("Simplified model is invalid!")

    # ================================================================
    # 阶段3: 统计简化后模型的参数量
    # ================================================================
    onnx_model_path = simplified_onnx_file_path
    total_params = count_parameters(onnx_model_path)
    print(f"Total number of parameters in the model: {total_params}")


# ================================================================
# 💡 关键问题：能否将 models/ 下的 .pth 转为 ONNX 再转为 BIN？
# ================================================================
#
# 回答：可以，但需要满足以下要求：
#
# ┌─────────────────────────────────────────────────────────────────┐
# │                    Gate 模型 (best_gate_model.pth)              │
# ├─────────────────────────────────────────────────────────────────┤
# │                                                                 │
# │  ✅ PTH → ONNX：完全可以，但本脚本需要修改 3 处：              │
# │                                                                 │
# │  1. 不能用 timm 的 create_model()，需要用 torchvision 的       │
# │     mobilenet_v3_small()，因为训练时用的是 torchvision          │
# │                                                                 │
# │  2. 必须修改分类头使之与训练时一致：                            │
# │     model.classifier[3] = nn.Linear(576, 1)  # 1000→1         │
# │                                                                 │
# │  3. 必须加载训练好的权重：                                      │
# │     model.load_state_dict(torch.load("best_gate_model.pth"))   │
# │                                                                 │
# │  ✅ ONNX → BIN：可以，但需要在 Docker 工具链内执行             │
# │     使用 hb_mapper makertbin 命令                               │
# │                                                                 │
# │  ⚠️ 关键要求：                                                  │
# │  - opset_version 建议 11 或 12                                  │
# │  - 必须先 onnxsim 简化                                          │
# │  - 输入名建议用 "images"（与训练代码一致）                      │
# │  - ONNX 算子必须全部是 BPU 支持的算子                           │
# │    （可用 hb_mapper checker 检查算子兼容性）                    │
# │  - 需要准备 50~200 张校准图片用于 INT8 量化                     │
# │                                                                 │
# └─────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────┐
# │                    YOLO 模型 (best.pt)                          │
# ├─────────────────────────────────────────────────────────────────┤
# │                                                                 │
# │  ⚠️ PTH → ONNX：不能用普通的 torch.onnx.export！               │
# │     必须使用 RDK Model Zoo 提供的 export_monkey_patch.py        │
# │     原因：官方 ultralytics 导出的 ONNX 只有 1 个输出头，       │
# │     而 RDK 推理代码需要 6 个输出头的格式                        │
# │                                                                 │
# │  ✅ ONNX → BIN：用 Docker 内的 mapper.py                       │
# │                                                                 │
# └─────────────────────────────────────────────────────────────────┘
#
# ─── 修改后的 Gate 导出代码示例 ───
#
#   import torch
#   import torch.nn as nn
#   from torchvision.models import mobilenet_v3_small
#   from onnxsim import simplify
#   import onnx
#
#   # 1. 重建模型架构（必须与训练时完全一致）
#   model = mobilenet_v3_small()
#   model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
#
#   # 2. 加载训练好的权重
#   model.load_state_dict(torch.load(
#       "../models/best_gate_model.pth", map_location="cpu"
#   ))
#   model.eval()
#
#   # 3. 导出 ONNX
#   dummy_input = torch.randn(1, 3, 224, 224)
#   torch.onnx.export(
#       model, dummy_input, "gate_mobilenetv3.onnx",
#       opset_version=11,
#       input_names=["images"],
#       output_names=["scores"],
#       do_constant_folding=True,
#   )
#
#   # 4. 简化 ONNX（RDK 工具链要求）
#   model_simp, check = simplify("gate_mobilenetv3.onnx")
#   if check:
#       onnx.save(model_simp, "gate_mobilenetv3_sim.onnx")
#
#   # 5. (在 Docker 工具链内) ONNX → BIN
#   #    hb_mapper makertbin \
#   #      --model-type onnx \
#   #      --onnx /data/gate_mobilenetv3_sim.onnx \
#   #      --model-name gate_mobilenetv3 \
#   #      --input-shape "images:1x3x224x224" \
#   #      --cal-image-dir /data/calibration_images \
#   #      --output-dir /data/gate_output
