from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.data import Folder
from anomalib.models import Fastflow
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

project_root = Path(__file__).resolve().parent
dataset_root = project_root / "Datasets" / "Maimu_AD"
output_root = project_root / "results" / "FastFlow" / "MaiMu_AD"
visual_dir = output_root / "visualizations"
checkpoint_path = output_root / "fastflow_resnet18.ckpt"

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

def build_model():
    return Fastflow(
        backbone="resnet18",
        pre_trained=True,
        evaluator=False,
        pre_processor=PreProcessor(transform=transform),
        visualizer=False,
    )

def build_train_datamodule():
    return Folder(
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

def build_export_datamodule():
    return Folder(
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

