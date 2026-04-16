from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Fastflow
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage


# =========================
# 1. 基本路径和参数
# =========================
project_root = Path(__file__).resolve().parent
dataset_root = project_root / "Datasets" / "Maimu_AD"
output_root = project_root / "results" 
visual_dir = output_root / "visualizations"
checkpoint_path = output_root / "fastflow_resnet18.ckpt"

image_size = 512
export_size = 1024
normal_visual_count = 8


# =========================
# 2. 预处理、模型、数据、引擎
# =========================
transform = Compose(
    [
        ToImage(),
        Resize((image_size, image_size), antialias=True),
        ToDtype(torch.float32, scale=True),
    ]
)

model = Fastflow(
    backbone="resnet18",
    pre_trained=True,
    evaluator=False,
    pre_processor=PreProcessor(transform=transform),
    visualizer=False,
)

train_datamodule = Folder(
    name="MaiMu_AD",
    root=dataset_root,
    normal_dir="good",
    abnormal_dir="bad",
    normal_split_ratio=0.2,
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=2,
    seed=42,
)

engine = Engine(
    max_epochs=30,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    default_root_dir=output_root,
    enable_checkpointing=True,
    limit_val_batches=0,
    logger=True,
    log_every_n_steps=1,
)


# =========================
# 3. 一些小工具函数
# =========================
def to_numpy(x):
    """把 tensor 转成 numpy，方便后面用 OpenCV 处理。"""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def normalize_map(anomaly_map):
    """把 anomaly map 归一化到 0~1。"""
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    anomaly_map = np.nan_to_num(anomaly_map, nan=0.0, posinf=0.0, neginf=0.0)

    while anomaly_map.ndim > 2:
        anomaly_map = anomaly_map[0]

    min_value = float(anomaly_map.min())
    max_value = float(anomaly_map.max())

    if max_value - min_value < 1e-8:
        return np.zeros_like(anomaly_map, dtype=np.float32)

    return (anomaly_map - min_value) / (max_value - min_value)


def evenly_sample_paths(paths, sample_count):
    """从正常样本里均匀抽几张做可视化。"""
    if sample_count <= 0 or len(paths) <= sample_count:
        return list(paths)

    indices = np.linspace(0, len(paths) - 1, sample_count, dtype=int)
    return [paths[i] for i in indices]


def collect_target_paths():
    """收集要导出的图片路径。坏样本全要，正常样本抽一部分。"""
    bad_paths = sorted([p for p in (dataset_root / "bad").glob("*") if p.is_file()])
    good_paths = sorted([p for p in (dataset_root / "good").glob("*") if p.is_file()])

    sampled_good_paths = evenly_sample_paths(good_paths, normal_visual_count)
    target_paths = set(bad_paths + sampled_good_paths)

    print(f"坏样本可视化: {len(bad_paths)} 张")
    print(f"正常样本可视化: {len(sampled_good_paths)} 张（从 {len(good_paths)} 张中抽样）")

    return target_paths, set(bad_paths), set(sampled_good_paths)


# =========================
# 4. 可视化相关函数
# =========================
def make_binary_mask(anomaly_map, is_bad_image):
    """
    把 anomaly map 变成二值区域。
    正常图直接返回空白 mask。
    """
    target_size = (export_size, export_size)

    if not is_bad_image:
        return np.zeros(target_size, dtype=np.uint8)

    threshold = max(0.55, float(np.quantile(anomaly_map, 0.985)))
    binary = (anomaly_map >= threshold).astype(np.uint8) * 255

    if np.count_nonzero(binary) > 0:
        return binary

    threshold = max(0.4, float(np.quantile(anomaly_map, 0.95)))
    return (anomaly_map >= threshold).astype(np.uint8) * 255


def save_comparison_image(image_path, anomaly_map, score, is_bad_image):
    """保存原图、热力图、叠加图三联图。"""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图像: {image_path}")

    target_size = (export_size, export_size)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    anomaly_map = normalize_map(anomaly_map)
    anomaly_map = cv2.resize(anomaly_map, target_size, interpolation=cv2.INTER_CUBIC)

    heatmap = cv2.applyColorMap((anomaly_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    binary_mask = make_binary_mask(anomaly_map, is_bad_image)
    kernel = np.ones((7, 7), dtype=np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    overlay = cv2.addWeighted(image, 0.55, heatmap, 0.45, 0)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(64, int(target_size[0] * target_size[1] * 0.0001))

    region_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        cv2.drawContours(overlay, [contour], -1, (0, 255, 255), 3)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 3)
        region_count += 1

    status_text = "Anomaly" if is_bad_image else "Normal"
    status_color = (0, 128, 255) if is_bad_image else (120, 255, 120)

    cv2.putText(overlay, f"score: {float(score):.4f}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.putText(overlay, status_text, (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3, cv2.LINE_AA)

    if is_bad_image:
        cv2.putText(overlay, f"regions: {region_count}", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    def add_title(img, title):
        title_height = 56
        canvas = np.full((img.shape[0] + title_height, img.shape[1], 3), 18, dtype=np.uint8)
        canvas[title_height:] = img
        cv2.putText(canvas, title, (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2, cv2.LINE_AA)
        return canvas

    comparison = np.concatenate(
        [
            add_title(image, "Original"),
            add_title(heatmap, "Anomaly Map"),
            add_title(overlay, "Overlay"),
        ],
        axis=1,
    )

    visual_dir.mkdir(parents=True, exist_ok=True)
    save_path = visual_dir / f"{image_path.parent.name}_{image_path.stem}_comparison.png"
    cv2.imwrite(str(save_path), comparison)
    return save_path


# =========================
# 5. 训练 / 保存
# =========================
def train_and_save():
    print("=== 开始训练 FastFlow(resnet18) ===")
    engine.fit(model=model, datamodule=train_datamodule)

    # print("=== 开始测试 ===")
    # engine.test(model=model, datamodule=train_datamodule)

    if engine.trainer is None:
        raise RuntimeError("trainer 未初始化，无法保存 checkpoint")

    engine.trainer.save_checkpoint(str(checkpoint_path))
    print(f"模型已保存: {checkpoint_path}")


# =========================
# 6. 预测 / 可视化
# =========================
def visualize_from_checkpoint():
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

    print("=== 开始导出 comparison.png ===")
    visual_dir.mkdir(parents=True, exist_ok=True)

    for old_file in visual_dir.glob("*_comparison.png"):
        old_file.unlink()

    target_paths, bad_targets, normal_targets = collect_target_paths()

    export_datamodule = Folder(
        name="MaiMu_AD_export",
        root=dataset_root,
        normal_dir="good",
        abnormal_dir="bad",
        normal_split_ratio=0.2,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=2,
        seed=42,
    )

    export_datamodule.setup()
    test_loader = export_datamodule.test_dataloader()
    if test_loader is None:
        raise RuntimeError("导出阶段未能创建 test_dataloader")

    predictions = engine.predict(
        model=model,
        dataloaders=test_loader,
        ckpt_path=str(checkpoint_path),
        return_predictions=True,
    )

    saved_paths = set()
    saved_count = 0

    for batch in predictions:
        image_paths = batch.image_path
        anomaly_maps = batch.anomaly_map
        scores = batch.pred_score

        for i in range(len(image_paths)):
            image_path = Path(image_paths[i])

            if image_path not in target_paths:
                continue
            if image_path in saved_paths:
                continue

            save_path = save_comparison_image(
                image_path=image_path,
                anomaly_map=to_numpy(anomaly_maps[i]),
                score=float(to_numpy(scores[i]).reshape(-1)[0]),
                is_bad_image=image_path in bad_targets,
            )

            print(f"已保存: {save_path}")
            saved_paths.add(image_path)
            saved_count += 1

    missing_paths = sorted(target_paths - saved_paths)

    print(f"目标导出共 {len(target_paths)} 张，已导出 {saved_count} 张。")
    print(f"其中坏样本目标 {len(bad_targets)} 张，正常样本目标 {len(normal_targets)} 张。")

    if missing_paths:
        print("以下图片未进入导出流程:")
        for path in missing_paths:
            print(f"  - {path}")


# =========================
# 7. 主入口
# =========================
if __name__ == "__main__":
    mode = "vis"   # 可改成 "train" 或 "visualize"

    if mode == "train":
        train_and_save()
    elif mode == "vis":
        visualize_from_checkpoint()
    else:
        raise ValueError("mode 只能是 'train' 或 'visualize'")