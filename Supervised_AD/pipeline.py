from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import mobilenet_v3_small
from torchvision.ops import nms

# Why: 延迟导入 ultralytics，允许仅使用 Gate 推理的场景无需安装该依赖
try:
    from ultralytics import YOLO
except ImportError as exc:
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None

# Why: 224 是 MobileNetV3 官方推荐输入尺寸，变更会降低 Gate 精度
MODEL_INPUT_SIZE = 224
# Why: 640 匹配 YOLO 训练时的输入尺寸，保证检测精度
PATCH_SIZE = 640
# Why: stride == PATCH_SIZE 表示切片无重叠；若需捕获边界处缺陷可减小此值
PATCH_STRIDE = 640
IMAGE_SIZE = 2048

# Why: ImageNet 标准归一化参数，必须与 Gate 模型训练时的预处理完全一致
GATE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
GATE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Watch: 硬编码绝对路径，换机器需手动修改
DATA_ROOT = Path("/home/wapiti/Projects/Datasets/MaiMu")
IMG_DIR = DATA_ROOT / "images"


@dataclass
class PatchInfo:
    patch_id: int
    x0: int
    y0: int
    width: int
    height: int
    image: Image.Image


@dataclass
class Detection:
    patch_id: int
    cls_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class GateYoloPipeline:
    """两阶段异常检测流水线：Gate 轻量分类器初筛 → YOLO 精细定位。

    设计动机：大图(2048x2048)直接送 YOLO 开销大且小缺陷易丢失，
    先用轻量 Gate 对 4x4 切片打分，仅对高风险切片运行 YOLO。
    """

    # Watch: gate_threshold 默认值在 __init__(0.1) 和 CLI(0.5) 中不一致，
    #        命令行未传 --gate-threshold 时实际使用 0.5
    def __init__(
        self,
        gate_model_path: str | Path,
        yolo_model_path: str | Path,
        device: str | None = None,
        gate_threshold: float = 0.1,
        yolo_conf: float = 0.25,
        yolo_iou: float = 0.45,
        merge_iou: float = 0.5,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gate_threshold = gate_threshold
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        self.merge_iou = merge_iou

        self.gate_model = self._load_gate_model(Path(gate_model_path))
        self.yolo_model = self._load_yolo_model(Path(yolo_model_path))

    def _load_gate_model(self, model_path: Path) -> nn.Module:
        if not model_path.exists():
            raise FileNotFoundError(f"Gate model not found: {model_path}")

        # Why: torch.load 只恢复权重张量，不保存模型结构，必须先重建与训练时一致的架构
        model = mobilenet_v3_small()
        # Watch: 分类头修改必须与 train_gate_rdk.py 中训练时一致（1000→1），
        #        否则权重形状不匹配会直接报错
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        return model

    def _load_yolo_model(self, model_path: Path) -> Any:
        if YOLO is None:
            raise ImportError("ultralytics is not installed.") from ULTRALYTICS_IMPORT_ERROR
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        return YOLO(str(model_path))

    def _build_patch_positions(self, image_width: int, image_height: int) -> list[tuple[int, int]]:
        xs = self._axis_positions(image_width)
        ys = self._axis_positions(image_height)
        return [(x0, y0) for y0 in ys for x0 in xs]

    @staticmethod
    def _axis_positions(length: int) -> list[int]:
        # Why: 采用"均匀三步 + 尾部对齐"策略，确保图像右/下边缘不被遗漏
        #      2048 尺寸下生成 [0, 640, 1280, 1408] 四个位置，1408 保证覆盖最右侧区域
        positions = [0, PATCH_STRIDE, PATCH_STRIDE * 2, max(length - PATCH_SIZE, 0)]
        unique_positions: list[int] = []
        for pos in positions:
            if pos not in unique_positions:
                unique_positions.append(pos)
        return unique_positions

    def split_patches(self, image: Image.Image) -> list[PatchInfo]:
        image = image.convert("RGB")
        patches: list[PatchInfo] = []
        for patch_id, (x0, y0) in enumerate(self._build_patch_positions(*image.size)):
            patch = image.crop((x0, y0, x0 + PATCH_SIZE, y0 + PATCH_SIZE))
            patches.append(
                PatchInfo(
                    patch_id=patch_id, x0=x0, y0=y0, width=PATCH_SIZE, height=PATCH_SIZE, image=patch
                )
            )
        return patches

    def build_gate_batch(self, patches: list[PatchInfo]) -> torch.Tensor:
        tensors = []
        for patch in patches:
            resized = patch.image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.BILINEAR)
            array = np.asarray(resized, dtype=np.float32) / 255.0
            # Why: 归一化参数必须与训练时一致，否则 Gate 输出分布偏移导致阈值失效
            array = (array - GATE_MEAN) / GATE_STD
            # Why: PyTorch 期望 CHW 格式，而 PIL 返回 HWC
            tensors.append(array.transpose(2, 0, 1))
        batch = torch.from_numpy(np.stack(tensors)).to(self.device)
        return batch

    @torch.no_grad()
    def run_gate(self, patches: list[PatchInfo]) -> tuple[list[float], list[int]]:
        batch = self.build_gate_batch(patches)
        logits = self.gate_model(batch).squeeze(-1)
        # Why: 模型输出原始 logit，需 sigmoid 转为 [0,1] 概率才能与 gate_threshold 比较
        scores = torch.sigmoid(logits).cpu().numpy().tolist()
        active_indices = [idx for idx, score in enumerate(scores) if score >= self.gate_threshold]
        return scores, active_indices

    def run_yolo(self, patches: list[PatchInfo], active_indices: list[int]) -> list[Detection]:
        if not active_indices:
            return []

        active_patches = [patches[idx] for idx in active_indices]
        # Why: 直接传 PIL Image 而非 numpy 数组，避免手动转换时 BGR/RGB 通道顺序错乱，
        #      ultralytics 内部对 PIL 输入有成熟的 RGB 处理路径
        pil_images = [patch.image for patch in active_patches]

        results = self.yolo_model.predict(
            source=pil_images,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            verbose=False,
            device=self.device,
        )

        detections: list[Detection] = []
        for patch, result in zip(active_patches, results):
            boxes = result.boxes
            if boxes is None or boxes.xyxy.numel() == 0:
                continue

            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            clss = boxes.cls.detach().cpu().numpy().astype(int)

            for box, conf, cls_id in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = box.tolist()
                # Why: YOLO 输出的是 patch 局部坐标，需加上 patch 偏移量还原图像全局坐标
                detections.append(
                    Detection(
                        patch_id=patch.patch_id,
                        cls_id=int(cls_id),
                        confidence=float(conf),
                        x1=float(x1 + patch.x0),
                        y1=float(y1 + patch.y0),
                        x2=float(x2 + patch.x0),
                        y2=float(y2 + patch.y0),
                    )
                )
        return detections

    def draw_visualization(
        self,
        image: Image.Image,
        patches: list[PatchInfo],
        active_indices: list[int],
        gate_scores: list[float],
        detections: list[Detection],
        output_path: str
    ) -> None:
        """可视化：半透明切片状态 (橙/蓝) + YOLO 预测框 (红)"""
        # Why: 用独立 RGBA 图层叠加，避免直接在原图上绘制时破坏像素
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Watch: 字体加载有三级回退，不同 Linux 发行版字体路径不同
        font_size = 36
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except OSError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()

        for patch in patches:
            x0, y0 = patch.x0, patch.y0
            x1, y1 = x0 + patch.width, y0 + patch.height
            score = gate_scores[patch.patch_id]
            is_pass = patch.patch_id in active_indices

            # Why: 橙色 = 有风险(送 YOLO)，蓝色 = 安全(跳过)，透明度低以不遮挡原图
            color = (255, 165, 0, 60) if is_pass else (100, 149, 237, 40)
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(color[:3] + (200,)), width=3)

            txt = f"ID:{patch.patch_id} S:{score:.3f}"
            draw.text((x0 + 10, y1 - 35), txt, fill="white", font=font)

        for det in detections:
            # Why: 红色不透明边框突出 YOLO 检测到的异常区域
            draw.rectangle([det.x1, det.y1, det.x2, det.y2], outline=(255, 0, 0, 255), width=6)

            label = f"Cls:{det.cls_id} {det.confidence:.2f}"
            # Watch: 标签置于框上方，若框贴近顶边则 label_y 会被 clamp 到 0
            label_y = max(det.y1 - font_size - 4, 0)
            bbox = draw.textbbox((det.x1, label_y), label, font=font)
            draw.rectangle(bbox, fill=(255, 0, 0, 255))
            draw.text((det.x1, label_y), label, fill="white", font=font)

        # Why: alpha_composite 将半透明叠加层与原图合成，再转回 RGB 以保存为 JPEG
        canvas = image.convert("RGBA")
        out_img = Image.alpha_composite(canvas, overlay).convert("RGB")
        out_img.save(output_path)
        print(f"可视化结果已保存至: {output_path}")

    def predict(self, image_path: str | Path, save_vis_path: str | None = None) -> dict[str, Any]:
        image_path = Path(IMG_DIR) / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path).convert("RGB") as image:
            patches = self.split_patches(image)
            gate_scores, active_indices = self.run_gate(patches)
            raw_detections = self.run_yolo(patches, active_indices)

            if save_vis_path:
                self.draw_visualization(image, patches, active_indices, gate_scores, raw_detections, save_vis_path)

            # Why: 按 patch_id 分组检测结果，方便下游按切片定位异常
            patch_predictions = {patch_id: [] for patch_id in active_indices}
            for det in raw_detections:
                patch_predictions[det.patch_id].append(asdict(det))

        return {
            "image_path": str(image_path),
            "gate_threshold": self.gate_threshold,
            "active_patches_count": len(active_indices),
            "patch_predictions": patch_predictions,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="ColorStreak_93.bmp")
    parser.add_argument("--gate-model", type=str, default="./models/best_gate_model.pth")
    parser.add_argument("--yolo-model", type=str, default="./models/best.pt")
    # Watch: 默认 0.5 与 GateYoloPipeline.__init__ 的默认 0.1 不同，
    #        这是有意为之：CLI 场景下 0.5 误报更少，程序化调用时 0.1 召回更高
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--save-vis", type=str, default="./vis_result.jpg", help="可视化图像的保存路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = GateYoloPipeline(
        gate_model_path=args.gate_model,
        yolo_model_path=args.yolo_model,
        device=args.device,
        gate_threshold=args.gate_threshold,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
    )

    result = pipeline.predict(args.image, save_vis_path=args.save_vis)

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
