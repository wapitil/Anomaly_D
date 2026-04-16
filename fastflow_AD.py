from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Fastflow
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage


# 固定数据和输出路径，直接运行即可。
project_root = Path(__file__).resolve().parent
dataset_root = project_root / "Datasets" / "Maimu_AD"
output_root = project_root / "results" / "FastFlow" / "MaiMu_AD"
visual_dir = output_root / "visualizations"
checkpoint_path = output_root / "fastflow_resnet18.ckpt"

# 训练输入保持 512，导出可视化提升到 1024，便于查看细节。
image_size = 512
export_size = 1024
normal_visual_count = 8

transform = Compose(
    [
        ToImage(),
        Resize((image_size, image_size), antialias=True),
        ToDtype(torch.float32, scale=True),
    ]
)

datamodule = Folder(
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

model = Fastflow(
    backbone="resnet18",
    pre_trained=True,
    evaluator=False,
    pre_processor=PreProcessor(transform=transform),
    visualizer=False,
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


def to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_map(anomaly_map):
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    anomaly_map = np.nan_to_num(anomaly_map, nan=0.0, posinf=0.0, neginf=0.0)
    while anomaly_map.ndim > 2:
        anomaly_map = anomaly_map[0]
    low = float(anomaly_map.min())
    high = float(anomaly_map.max())
    if high - low < 1e-8:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    return (anomaly_map - low) / (high - low)


def evenly_sample_paths(paths, sample_count):
    if sample_count <= 0 or len(paths) <= sample_count:
        return list(paths)
    indices = np.linspace(0, len(paths) - 1, sample_count, dtype=int)
    return [paths[idx] for idx in indices]


def collect_export_targets():
    bad_paths = sorted(path for path in (dataset_root / "bad").glob("*") if path.is_file())
    good_paths = sorted(path for path in (dataset_root / "good").glob("*") if path.is_file())
    sampled_good_paths = evenly_sample_paths(good_paths, normal_visual_count)
    target_paths = set(bad_paths + sampled_good_paths)

    print(f"坏样本可视化: {len(bad_paths)} 张")
    print(f"正常样本可视化: {len(sampled_good_paths)} 张（从 {len(good_paths)} 张中抽样）")
    return target_paths, set(bad_paths), set(sampled_good_paths)


def get_field(item, field_name):
    if item is None:
        return None
    if isinstance(item, dict):
        return item.get(field_name)
    return getattr(item, field_name, None)


def is_prediction_container(item):
    return get_field(item, "image_path") is not None and get_field(item, "anomaly_map") is not None


def flatten_prediction_items(predictions):
    if predictions is None:
        return
    if is_prediction_container(predictions):
        yield predictions
        return
    if isinstance(predictions, (list, tuple)):
        for value in predictions:
            yield from flatten_prediction_items(value)


def normalize_image_paths(image_paths):
    if image_paths is None:
        return []
    if isinstance(image_paths, (str, Path)):
        return [Path(image_paths)]
    return [Path(path) for path in image_paths]


def split_batch_field(value, batch_size):
    if batch_size == 0:
        return []
    if value is None:
        return [None] * batch_size
    if isinstance(value, (list, tuple)):
        if len(value) == batch_size:
            return list(value)
        if len(value) == 1 and batch_size > 1:
            return list(value) * batch_size
        return list(value[:batch_size])
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return [value[index] for index in range(batch_size)]
        return [value] * batch_size

    array = np.asarray(value)
    if array.ndim > 0 and array.shape[0] == batch_size:
        return [array[index] for index in range(batch_size)]
    return [value] * batch_size


def build_binary_mask(anomaly_map, pred_mask, is_anomaly):
    target_size = (export_size, export_size)
    if not is_anomaly:
        return np.zeros(target_size, dtype=np.uint8)

    if pred_mask is not None:
        mask = to_numpy(pred_mask)
        while mask.ndim > 2:
            mask = mask[0]
        mask = cv2.resize(mask.astype(np.float32), target_size, interpolation=cv2.INTER_NEAREST)
        binary = (mask > 0).astype(np.uint8) * 255
        if np.count_nonzero(binary) > 0:
            return binary

    threshold = max(0.55, float(np.quantile(anomaly_map, 0.985)))
    binary = (anomaly_map >= threshold).astype(np.uint8) * 255
    if np.count_nonzero(binary) > 0:
        return binary

    threshold = max(0.4, float(np.quantile(anomaly_map, 0.95)))
    return (anomaly_map >= threshold).astype(np.uint8) * 255


def save_comparison(image_path, anomaly_map, pred_mask, score, is_anomaly):
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise RuntimeError(f"无法读取图像: {image_path}")

    target_size = (export_size, export_size)
    original = cv2.resize(original, target_size, interpolation=cv2.INTER_AREA)

    anomaly_map = normalize_map(anomaly_map)
    anomaly_map = cv2.resize(anomaly_map, target_size, interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap((anomaly_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    binary = build_binary_mask(anomaly_map, pred_mask, is_anomaly)
    kernel = np.ones((7, 7), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    overlay = cv2.addWeighted(original, 0.55, heatmap, 0.45, 0)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(64, int(target_size[0] * target_size[1] * 0.0001))
    contour_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        cv2.drawContours(overlay, [contour], -1, (0, 255, 255), 3)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 3)
        contour_count += 1

    status_text = "Anomaly" if is_anomaly else "Normal"
    status_color = (0, 128, 255) if is_anomaly else (120, 255, 120)
    cv2.putText(
        overlay,
        status_text,
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        status_color,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"score: {float(score):.4f}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    if is_anomaly:
        cv2.putText(
            overlay,
            f"regions: {contour_count}",
            (30, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    title_h = 56

    def add_title(image, title):
        canvas = np.full((image.shape[0] + title_h, image.shape[1], 3), 18, dtype=np.uint8)
        canvas[title_h:] = image
        cv2.putText(
            canvas,
            title,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        return canvas

    comparison = np.concatenate(
        [
            add_title(original, "Original"),
            add_title(heatmap, "Anomaly Map"),
            add_title(overlay, "Overlay"),
        ],
        axis=1,
    )

    visual_dir.mkdir(parents=True, exist_ok=True)
    save_path = visual_dir / f"{image_path.parent.name}_{image_path.stem}_comparison.png"
    cv2.imwrite(str(save_path), comparison)
    return save_path


def export_visualizations(predictions, target_paths, abnormal_targets):
    saved_paths = set()
    saved_count = 0

    for item in flatten_prediction_items(predictions):
        image_paths = normalize_image_paths(get_field(item, "image_path"))
        if not image_paths:
            continue

        batch_size = len(image_paths)
        anomaly_maps = split_batch_field(get_field(item, "anomaly_map"), batch_size)
        pred_masks = split_batch_field(get_field(item, "pred_mask"), batch_size)
        scores = split_batch_field(get_field(item, "pred_score"), batch_size)

        for index, image_path in enumerate(image_paths):
            if image_path not in target_paths or image_path in saved_paths:
                continue

            save_path = save_comparison(
                image_path=image_path,
                anomaly_map=anomaly_maps[index],
                pred_mask=pred_masks[index],
                score=float(np.asarray(scores[index]).reshape(-1)[0]),
                is_anomaly=image_path in abnormal_targets,
            )
            print(f"已保存: {save_path}")
            saved_paths.add(image_path)
            saved_count += 1

    return saved_count, saved_paths


print("=== 开始训练 FastFlow(resnet18) ===")
engine.fit(model=model, datamodule=datamodule)

print("=== 开始测试 ===")
engine.test(model=model, datamodule=datamodule)

if engine.trainer is None:
    raise RuntimeError("trainer 未初始化")
engine.trainer.save_checkpoint(str(checkpoint_path))
print(f"模型已保存: {checkpoint_path}")

print("=== 开始导出 comparison.png ===")
visual_dir.mkdir(parents=True, exist_ok=True)
for old_file in visual_dir.glob("*_comparison.png"):
    old_file.unlink()

target_paths, abnormal_targets, normal_targets = collect_export_targets()

export_datamodule = Folder(
    name="MaiMu_AD_export",
    root=dataset_root,
    normal_dir="good",
    abnormal_dir="bad",
    normal_test_dir="good",
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

saved_count, saved_paths = export_visualizations(predictions, target_paths, abnormal_targets)
missing_paths = sorted(target_paths - saved_paths)

print(f"目标导出共 {len(target_paths)} 张，已导出 {saved_count} 张。")
print(f"其中坏样本目标 {len(abnormal_targets)} 张，正常样本目标 {len(normal_targets)} 张。")
if missing_paths:
    print("以下图片未进入导出流程:")
    for path in missing_paths:
        print(f"  - {path}")
