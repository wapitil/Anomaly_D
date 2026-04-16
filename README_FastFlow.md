# FastFlow 异常检测训练说明

本目录提供基于 anomalib 2.x 的 FastFlow 训练脚本，使用 `resnet18` 骨干网络，适配如下数据结构：

```text
Datasets/Maimu_AD/
├── good/
└── test/
```

其中：

- `good/` 用于训练正常样本
- `test/` 用于推理和评估
- 不使用 mask 图像

## 脚本

训练与推理脚本：

```bash
python fastflow_AD.py
```

常用参数：

```bash
python fastflow_AD.py \
  --dataset-root Datasets/Maimu_AD \
  --output-root results/FastFlow/MaiMu_AD \
  --epochs 30 \
  --image-size 512 \
  --train-batch-size 4 \
  --eval-batch-size 1 \
  --num-workers 0 \
  --device auto
```

## 输出结果

默认输出目录：

```text
results/FastFlow/MaiMu_AD/
├── fastflow_resnet18.ckpt
├── summary.json
└── visualizations/
    └── xxx_comparison.png
```

其中：

- `comparison.png`：原图、热力图、叠加图、轮廓图、二值掩膜五联图
- `summary.json`：测试指标与导出结果汇总
- 如需额外导出 `overlay/heatmap/mask`，可追加 `--export-extra-views`

## 依赖安装

建议使用 Python 3.10 到 3.12。

```bash
pip install "anomalib[core]"
pip install opencv-python pillow
```

如果要用 GPU，请额外安装与你 CUDA 版本匹配的 PyTorch。
