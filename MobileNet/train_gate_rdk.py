from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

DATA_ROOT = Path("/home/wapiti/Projects/Datasets/MaiMu_Patches")
IMG_DIR = DATA_ROOT / "images"
LBL_DIR = DATA_ROOT / "labels"
OUT_DIR = Path("/home/wapiti/Projects/Anomaly_D/vis_outputs")


IMAGE_SIZE = 2048
PATCH_SIZE = 640
PATCH_POSITIONS = [0, 640, 1280, IMAGE_SIZE - PATCH_SIZE]
MODEL_INPUT_SIZE = 224  #

SEED = 42
LR = 1e-3
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TARGET_RECALL = 0.98


@dataclass
class PatchSample:
    image_id: int
    image_path: str
    patch_id: int
    x0: int
    y0: int
    label: int


def set_seed(seed: int) -> None:
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def yolo_to_xyxy(line: str) -> tuple[int, tuple[float, float, float, float]]:
    """将YOLO格式的标注(归一化坐标)转换为绝对像素坐标的(x1,y1,x2,y2)格式"""
    cls_id, x, y, w, h = line.split()
    cx = float(x) * IMAGE_SIZE
    cy = float(y) * IMAGE_SIZE
    bw = float(w) * IMAGE_SIZE
    bh = float(h) * IMAGE_SIZE
    return int(cls_id), (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def intersection_area(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """计算两个矩形框的交集面积"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    return inter_w * inter_h



class PatchDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples
        # 预定义标准化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        # 直接读取预裁切好的 224x224 小图
        with Image.open(s["patch_path"]) as img:
            array = np.asarray(img, dtype=np.float32) / 255.0
            
        tensor = torch.from_numpy(array.transpose(2, 0, 1))
        tensor = (tensor - self.mean) / self.std
        
        return (
            tensor,
            torch.tensor(float(s["label"]), dtype=torch.float32),
            # 为了兼容你之前的指标计算代码，可以传占位符或简化的 ID
            torch.tensor(index, dtype=torch.long), 
            torch.tensor(index, dtype=torch.long)
        )


def split_by_image(
    samples: list[dict],  # 修改类型注解
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """按图像ID划分数据集（支持字典格式）"""
    # 将 sample.image_id 修改为 sample['image_id']
    image_ids = sorted({sample['image_id'] for sample in samples})
    random.Random(SEED).shuffle(image_ids)

    train_end = int(len(image_ids) * train_ratio)
    val_end = train_end + int(len(image_ids) * val_ratio)

    train_ids = set(image_ids[:train_end])
    val_ids = set(image_ids[train_end:val_end])
    
    # 同样修改此处的访问方式
    train_samples = [sample for sample in samples if sample['image_id'] in train_ids]
    val_samples = [sample for sample in samples if sample['image_id'] in val_ids]
    test_samples = [
        sample
        for sample in samples
        if sample['image_id'] not in train_ids and sample['image_id'] not in val_ids
    ]
    return train_samples, val_samples, test_samples


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    total_epochs: int
) -> float:
    model.train()
    running_loss = 0.0
    # 使用 tqdm 包裹 loader，增加进度条显示
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    
    for images, labels, _, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        logits = model(images).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    return running_loss / len(loader)



def patch_metrics(
    records: list[tuple[int, int, int, float]], threshold: float
) -> dict[str, float]:
    """计算patch级别的分类指标：精确率、召回率、特异度、F1分数等"""
    labels = np.asarray([item[2] for item in records], dtype=np.int64)
    preds = np.asarray(
        [1 if item[3] >= threshold else 0 for item in records], dtype=np.int64
    )
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def image_metrics(
    records: list[tuple[int, int, int, float]], threshold: float
) -> dict[str, float]:
    """计算图像级别的评估指标：Top-1命中率、Top-2命中率、平均YOLO调用次数等"""
    grouped: dict[int, list[tuple[int, int, float]]] = {}
    for image_id, patch_id, label, score in records:
        grouped.setdefault(image_id, []).append((patch_id, label, score))

    top1_hit = 0
    top2_hit = 0
    avg_calls = 0.0
    avg_positive = 0.0

    for rows in grouped.values():
        rows.sort(key=lambda item: item[2], reverse=True)
        positive_patch_ids = {patch_id for patch_id, label, _ in rows if label == 1}
        avg_positive += len(positive_patch_ids)
        if rows[0][0] in positive_patch_ids:
            top1_hit += 1
        if any(row[0] in positive_patch_ids for row in rows[:2]):
            top2_hit += 1
        selected = sum(1 for _, _, score in rows if score >= threshold)
        avg_calls += max(selected, 1)

    num_images = max(len(grouped), 1)
    return {
        "num_images": num_images,
        "top1_hit_rate": top1_hit / num_images,
        "top2_hit_rate": top2_hit / num_images,
        "avg_yolo_calls": avg_calls / num_images,
        "avg_positive_patches": avg_positive / num_images,
    }

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float().view(-1, 1)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        # 将输出概率转为 0 或 1 (阈值 0.5)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def choose_threshold_for_recall(
    records: list[tuple[int, int, int, float]], target_recall: float = TARGET_RECALL
) -> float:
    """根据目标召回率选择最佳阈值，在满足召回率要求的前提下最大化特异度"""
    scores = np.asarray([item[3] for item in records], dtype=np.float32)
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 199)))
    best_threshold = 0.5
    best_specificity = -1.0
    for threshold in candidates:
        metrics = patch_metrics(records, float(threshold))
        if (
            metrics["recall"] >= target_recall
            and metrics["specificity"] > best_specificity
        ):
            best_threshold = float(threshold)
            best_specificity = metrics["specificity"]
    return best_threshold


def save_records(records: list[tuple[int, int, int, float]], output_path: Path) -> None:
    """将推理结果保存为JSON文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"image_id": image_id, "patch_id": patch_id, "label": label, "score": score}
        for image_id, patch_id, label, score in records
    ]
    output_path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Train a lightweight patch gate for MaiMu on 640x640 tiles."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--target-recall", type=float, default=TARGET_RECALL)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    return parser.parse_args()



def main() -> None:
    """主函数：训练轻量级patch分类器，评估性能并保存模型"""
    args = parse_args()
    set_seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    METADATA_PATH = Path("/home/wapiti/Projects/Datasets/MaiMu_Patches/metadata.json")
    with open(METADATA_PATH, "r") as f:
        samples = json.load(f)
    train_samples, val_samples, test_samples = split_by_image(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    # 修改后的提取逻辑
    def get_unique_ids(sample_list):
        # 提取字典中的 image_id 并去重
        return sorted(list(set(s["image_id"] for s in sample_list)))

    train_names = get_unique_ids(train_samples)
    val_names = get_unique_ids(val_samples)
    test_names = get_unique_ids(test_samples)

    # 将结果保存到输出目录
    split_info = {
        "train": train_names,
        "val": val_names,
        "test": test_names
    }
    with open(OUT_DIR / "data_split_log.json", "w") as f:
        json.dump(split_info, f, indent=4)

    print(f"device: {device}")
    print(f"total_samples: {len(samples)}")
    print(f"train_samples: {len(train_samples)}")
    print(f"val_samples: {len(val_samples)}")
    print(f"test_samples: {len(test_samples)}")

    train_positive = sum(sample["label"] for sample in train_samples)
    train_negative = len(train_samples) - train_positive
    pos_weight = torch.tensor(
        [train_negative / max(train_positive, 1)], dtype=torch.float32, device=device
    )
    print(f"train_positive: {train_positive}")
    print(f"train_negative: {train_negative}")
    print(f"pos_weight: {pos_weight.item():.4f}")

    train_loader = DataLoader(
        PatchDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        PatchDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        PatchDataset(test_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # 模型引入
    # model = MobileGateNet().to(device)
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    # 将分类头替换为二分类输出（1000 -> 1）
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0  # 初始化最高准确率记录
    for epoch in range(1, args.epochs + 1):
        # 1. 训练
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        
        # 2. 验证：每轮跑完检查一下“没见过”的数据
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # 3. 打印清晰的日志
        print(f"Epoch [{epoch:03d}/{args.epochs}] "
              f"TrainLoss: {train_loss:.4f} | "
              f"ValLoss: {val_loss:.4f} | "
              f"ValAcc: {val_acc:.2%}")

        # 4. 自动保存表现最好的权重
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUT_DIR / "best_gate_model.pth")
            print(f"  --> [Save] New best accuracy: {val_acc:.2%}")

    # 训练结束后，加载表现最好的权重进行最终评估
    model.load_state_dict(torch.load(OUT_DIR / "best_gate_model.pth"))
    print("\nTraining complete. Evaluating best model...")
    

    model.eval()
    dummy_input = torch.randn(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        str(OUT_DIR / "gate_tinynet_4x4.onnx"),
        input_names=["images"],
        output_names=["scores"],
        opset_version=11,
    )
    print(f"saved_onnx: {OUT_DIR / 'gate_tinynet_4x4.onnx'}")


if __name__ == "__main__":
    main()
