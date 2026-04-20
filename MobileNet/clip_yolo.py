import os
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ================= 配置区 =================
RAW_DATA_ROOT = Path("/home/wapiti/Projects/Datasets/MaiMu")
RAW_IMG_DIR = RAW_DATA_ROOT / "images"
RAW_LBL_DIR = RAW_DATA_ROOT / "labels"

YOLO_OUT_ROOT = Path("/home/wapiti/Projects/Datasets/MaiMu_YOLO")
# 所有的实体文件都存在这里
POOL_DIR = YOLO_OUT_ROOT / "all_data" 

# 参数保持一致
IMAGE_SIZE = 2048
PATCH_SIZE = 640
PATCH_POSITIONS = [0, 640, 1280, IMAGE_SIZE - PATCH_SIZE]
SEED = 42
TRAIN_RATIO = 0.8  # 80% 训练, 20% 验证
# ==========================================

def get_absolute_boxes(label_path):
    boxes = []
    if not label_path.exists(): return boxes
    for line in label_path.read_text().splitlines():
        if not line.strip(): continue
        cls, x, y, w, h = map(float, line.split())
        cx, cy = x * IMAGE_SIZE, y * IMAGE_SIZE
        bw, bh = w * IMAGE_SIZE, h * IMAGE_SIZE
        boxes.append([int(cls), cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2])
    return boxes

def convert_to_local(g_box, p_box):
    cls, gx1, gy1, gx2, gy2 = g_box
    px1, py1, px2, py2 = p_box
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    w_local, h_local = ix2 - ix1, iy2 - iy1
    cx_local = (ix1 + ix2) / 2 - px1
    cy_local = (iy1 + iy2) / 2 - py1
    return cls, cx_local / PATCH_SIZE, cy_local / PATCH_SIZE, w_local / PATCH_SIZE, h_local / PATCH_SIZE

def setup_yolo_folders():
    """创建YOLO标准目录结构"""
    for split in ["train", "val"]:
        (YOLO_OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
    (POOL_DIR / "images").mkdir(parents=True, exist_ok=True)
    (POOL_DIR / "labels").mkdir(parents=True, exist_ok=True)

def main():
    setup_yolo_folders()
    
    # 1. 划分大图 ID（确保同一大图的所有切片都在一起）
    all_stems = sorted([f.stem for f in RAW_LBL_DIR.glob("*.txt")])
    random.seed(SEED)
    random.shuffle(all_stems)
    
    split_idx = int(len(all_stems) * TRAIN_RATIO)
    train_stems = set(all_stems[:split_idx])
    
    # 2. 生成切片并存入数据池
    for stem in tqdm(all_stems, desc="处理切片"):
        lbl_path = RAW_LBL_DIR / f"{stem}.txt"
        img_path = RAW_IMG_DIR / f"{stem}.bmp"
        if not img_path.exists(): continue

        split = "train" if stem in train_stems else "val"
        global_boxes = get_absolute_boxes(lbl_path)
        
        with Image.open(img_path).convert("RGB") as full_img:
            p_idx = 0
            for y0 in PATCH_POSITIONS:
                for x0 in PATCH_POSITIONS:
                    p_box = (x0, y0, x0 + PATCH_SIZE, y0 + PATCH_SIZE)
                    local_labels = [convert_to_local(gb, p_box) for gb in global_boxes 
                                    if max(0, min(p_box[2], gb[3]) - max(p_box[0], gb[1])) * max(0, min(p_box[3], gb[4]) - max(p_box[1], gb[2])) > 0]

                    if local_labels:
                        patch_name = f"{stem}_p{p_idx}"
                        
                        # --- A. 写入实体文件到 POOL ---
                        img_save_path = POOL_DIR / "images" / f"{patch_name}.jpg"
                        lbl_save_path = POOL_DIR / "labels" / f"{patch_name}.txt"
                        
                        full_img.crop(p_box).save(img_save_path, quality=95)
                        txt_content = "\n".join([f"{l[0]} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}" for l in local_labels])
                        lbl_save_path.write_text(txt_content)

                        # --- B. 创建软连接到对应的 train/val ---
                        # 计算相对路径（为了以后移动文件夹模型还能找着）
                        ln_img = YOLO_OUT_ROOT / "images" / split / f"{patch_name}.jpg"
                        ln_lbl = YOLO_OUT_ROOT / "labels" / split / f"{patch_name}.txt"
                        
                        # 如果软连接已存在则先删除
                        if ln_img.is_symlink(): ln_img.unlink()
                        if ln_lbl.is_symlink(): ln_lbl.unlink()
                        
                        # 创建软连接，指向 POOL 里的实体
                        os.symlink(os.path.relpath(img_save_path, ln_img.parent), ln_img)
                        os.symlink(os.path.relpath(lbl_save_path, ln_lbl.parent), ln_lbl)

                    p_idx += 1

    print(f"\n[完成] 实体文件位于: {POOL_DIR}")
    print(f"[完成] YOLO 路径映射位于: {YOLO_OUT_ROOT}/images 和 labels")

if __name__ == "__main__":
    main()