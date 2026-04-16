import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 配置路径
RAW_DATA_ROOT = Path("/home/wapiti/Projects/Datasets/MaiMu")
RAW_IMG_DIR = RAW_DATA_ROOT / "images"
RAW_LBL_DIR = RAW_DATA_ROOT / "labels"

# 输出路径
PATCH_OUT_ROOT = Path("/home/wapiti/Projects/Datasets/MaiMu_Patches")
PATCH_IMG_DIR = PATCH_OUT_ROOT / "images"
PATCH_IMG_DIR.mkdir(parents=True, exist_ok=True)

# 参数保持一致
IMAGE_SIZE = 2048
PATCH_SIZE = 640
PATCH_POSITIONS = [0, 640, 1280, IMAGE_SIZE - PATCH_SIZE]

def pre_render_patches():
    label_files = sorted(RAW_LBL_DIR.glob("*.txt"))
    patch_metadata = []

    for label_path in tqdm(label_files, desc="Processing Images"):
        img_name = f"{label_path.stem}.bmp"
        img_path = RAW_IMG_DIR / img_name
        if not img_path.exists(): continue

        # 读取一次大图
        with Image.open(img_path) as full_img:
            full_img = full_img.convert("RGB")
            
            # 读取标注
            lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
            # 这里复用你之前的 yolo_to_xyxy 逻辑
            boxes = []
            for line in lines:
                cls_id, x, y, w, h = line.split()
                cx, cy = float(x) * IMAGE_SIZE, float(y) * IMAGE_SIZE
                bw, bh = float(w) * IMAGE_SIZE, float(h) * IMAGE_SIZE
                boxes.append((cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2))

            # 切片循环
            p_idx = 0
            for y0 in PATCH_POSITIONS:
                for x0 in PATCH_POSITIONS:
                    patch_box = (x0, y0, x0 + PATCH_SIZE, y0 + PATCH_SIZE)
                    
                    # 判断标签
                    has_defect = 0
                    for bx in boxes:
                        inter_w = max(0.0, min(patch_box[2], bx[2]) - max(patch_box[0], bx[0]))
                        inter_h = max(0.0, min(patch_box[3], bx[3]) - max(patch_box[1], bx[1]))
                        if (inter_w * inter_h) > 0:
                            has_defect = 1
                            break
                    
                    # 保存小图
                    patch_filename = f"{label_path.stem}_p{p_idx}.jpg"
                    patch_crop = full_img.crop(patch_box)
                    # 关键：存为训练尺寸 224x224，更进一步减轻训练 I/O
                    patch_crop.resize((224, 224), Image.BILINEAR).save(PATCH_IMG_DIR / patch_filename, quality=95)

                    patch_metadata.append({
                        "image_id": label_path.stem,
                        "patch_path": str(PATCH_IMG_DIR / patch_filename),
                        "label": has_defect
                    })
                    p_idx += 1

    # 保存元数据
    with open(PATCH_OUT_ROOT / "metadata.json", "w") as f:
        json.dump(patch_metadata, f, indent=2)

if __name__ == "__main__":
    pre_render_patches()