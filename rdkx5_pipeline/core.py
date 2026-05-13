from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models
from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    input_size: int
    feature_dim: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


BACKBONES = {
    "mobilenet_v2": BackboneConfig(
        name="mobilenet_v2",
        input_size=1024,
        feature_dim=1280,
        mean=tuple(float(x) for x in IMAGENET_MEAN),
        std=tuple(float(x) for x in IMAGENET_STD),
    ),
    "resnet18": BackboneConfig(
        name="resnet18",
        input_size=640,
        feature_dim=512,
        mean=tuple(float(x) for x in IMAGENET_MEAN),
        std=tuple(float(x) for x in IMAGENET_STD),
    ),
}


class FeatureBackbone(torch.nn.Module):
    def __init__(self, backbone_name: str) -> None:
        super().__init__()
        if backbone_name == "mobilenet_v2":
            model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            self.features = model.features
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif backbone_name == "resnet18":
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.features = torch.nn.Sequential(*list(model.children())[:-2])
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError(f"unsupported backbone: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


def get_backbone_config(backbone_name: str) -> BackboneConfig:
    if backbone_name not in BACKBONES:
        valid = ", ".join(BACKBONES)
        raise ValueError(f"backbone must be one of: {valid}")
    return BACKBONES[backbone_name]


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTS else []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def letterbox_resize_rgb(image_path: Path, input_size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    scale = min(input_size / width, input_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (input_size, input_size), (0, 0, 0))
    left = (input_size - resized_width) // 2
    top = (input_size - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def image_to_normalized_tensor(image_path: Path, input_size: int) -> torch.Tensor:
    image = letterbox_resize_rgb(image_path, input_size)
    array = np.asarray(image).astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, 1e-12)


def load_feature_model(backbone_name: str) -> torch.nn.Module:
    model = FeatureBackbone(backbone_name)
    return model.eval().to(DEVICE)


@torch.inference_mode()
def extract_float_features(
    image_paths: list[Path], backbone_name: str, batch_size: int
) -> np.ndarray:
    config = get_backbone_config(backbone_name)
    model = load_feature_model(backbone_name)
    all_features: list[np.ndarray] = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch = [
            image_to_normalized_tensor(path, config.input_size) for path in batch_paths
        ]
        tensor = torch.stack(batch).to(DEVICE)
        raw_features = model(tensor).cpu().numpy().astype(np.float32)
        all_features.append(l2_normalize(raw_features))

    return np.concatenate(all_features, axis=0)


def compute_stats(
    features: np.ndarray, threshold_quantile: float
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    center = features.mean(axis=0).astype(np.float32)
    scale = (features.std(axis=0) + 1e-6).astype(np.float32)
    scores = score_features(features, center, scale)
    threshold = float(np.quantile(scores, threshold_quantile))
    return center, scale, threshold, scores


def score_features(
    features: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    z = (features - center) / scale
    return np.sqrt(np.mean(z * z, axis=1))


def save_stats(
    output_path: Path,
    center: np.ndarray,
    scale: np.ndarray,
    threshold: float,
    backbone_name: str,
    source: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        center=center.astype(np.float32),
        scale=scale.astype(np.float32),
        threshold=np.array(threshold, dtype=np.float32),
        backbone=np.array(backbone_name),
        source=np.array(source),
    )


def load_stats(stats_path: Path) -> tuple[np.ndarray, np.ndarray, float, str]:
    data = np.load(stats_path)
    center = data["center"]
    scale = data["scale"]
    threshold = float(data["threshold"])
    backbone = str(data["backbone"]) if "backbone" in data.files else "mobilenet_v2"
    return center, scale, threshold, backbone


def save_metadata(
    output_path: Path, backbone_name: str, extra: dict[str, object]
) -> None:
    config = get_backbone_config(backbone_name)
    metadata = asdict(config)
    metadata.update(extra)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_feature_file(feature_path: Path) -> np.ndarray:
    if feature_path.suffix == ".npy":
        features = np.load(feature_path)
    elif feature_path.suffix == ".npz":
        data = np.load(feature_path)
        key = "features" if "features" in data.files else data.files[0]
        features = data[key]
    elif feature_path.suffix == ".csv":
        features = load_feature_csv(feature_path)
    else:
        raise ValueError("feature file must be .npy, .npz, or .csv")

    if features.ndim != 2:
        raise ValueError(f"features must be a 2D array, got shape {features.shape}")
    return features.astype(np.float32)


def load_feature_csv(feature_path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with feature_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            try:
                rows.append([float(value) for value in row])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no numeric feature rows found in {feature_path}")
    return np.array(rows, dtype=np.float32)
