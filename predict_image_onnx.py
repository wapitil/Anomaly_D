import cv2
import numpy as np
import onnxruntime

# 配置路径和阈值
IMAGE_PATH = "./Datasets/MVTecAD/leather/test/color/000.png"  # 输入图像路径
MODEL_PATH = "./Fastflow.onnx"  # ONNX模型路径
IMAGE_SIZE = 256  # 图像调整大小的目标尺寸
SCORE_THRESH = 0.5  # 异常图生成二值掩码的阈值
BIG_AREA_THRESH = 2000  # 将缺陷分类为“大”的阈值
MIN_AREA = 20  # 检测到的缺陷最小面积
PAD = 10  # 检测到的缺陷边界框的填充大小


def preprocess_for_model(image_bgr: np.ndarray) -> np.ndarray:
    # 预处理图像
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))  # 调整图像尺寸
    input_float = resized.astype(np.float32) / 255.0  # 标准化为浮动数据
    input_chw = np.transpose(input_float, (2, 0, 1))  # 转换为 NCHW 格式
    return np.expand_dims(input_chw, axis=0)


def run_onnx(input_nchw: np.ndarray) -> tuple[float, np.ndarray]:
    """
    执行 ONNX 模型推理
    参数:
        input_nchw: 预处理后的图像，NCHW 格式
    返回:
        包含两个值的元组:
            - score: 异常分数
            - anomaly_map: 2D 异常图
    """
    # 加载 ONNX 模型
    session = onnxruntime.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],  # 使用 CPU 进行推理
    )
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    # 执行推理
    output_values = session.run(output_names, {input_name: input_nchw})
    outputs = dict(zip(output_names, output_values))

    # 提取异常分数和异常图
    score = float(np.asarray(outputs["output"]).reshape(-1)[0])
    anomaly_map = np.asarray(outputs["anomaly_map"], dtype=np.float32)
    return score, anomaly_map


def make_mask(anomaly_map: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    """
    从异常图生成二值掩码
    参数:
        anomaly_map: 2D 异常图
        image_shape: 原始图像的形状 (H, W, C)
    返回:
        原始图像大小的二值掩码
    """
    # 压缩和归一化异常图
    map_2d = np.squeeze(anomaly_map).astype(np.float32)
    map_norm = (map_2d - float(map_2d.min())) / float(map_2d.max() - map_2d.min())
    # 应用阈值生成二值掩码
    mask_small = (map_norm >= SCORE_THRESH).astype(np.uint8) * 255
    # 调整掩码大小为原始图像尺寸
    h, w = image_shape[:2]
    return cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)


def split_defects_by_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """
    根据二值掩码分割大缺陷和小缺陷
    参数:
        image_bgr: 原始 BGR 图像
        mask: 缺陷的二值掩码
    返回:
        包含两个列表的元组:
            - big_defects: 大缺陷列表，每个缺陷是一个字典，包含 'area', 'bbox', 'roi'
            - small_defects: 小缺陷列表，每个缺陷是一个字典，包含 'area', 'bbox', 'roi'
    """
    # 查找掩码中的连接组件
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    big_defects: list[dict[str, object]] = []
    small_defects: list[dict[str, object]] = []
    h, w = image_bgr.shape[:2]

    # 遍历每个连接组件（跳过背景）
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_AREA:  # 跳过小的组件
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])

        # 计算带填充的边界框
        x1 = max(0, x - PAD)
        y1 = max(0, y - PAD)
        x2 = min(w, x + ww + PAD)
        y2 = min(h, y + hh + PAD)

        defect = {
            "area": area,
            "bbox": (x1, y1, x2, y2),
            "roi": image_bgr[y1:y2, x1:x2],  # 缺陷区域
        }
        if area >= BIG_AREA_THRESH:
            big_defects.append(defect)
        else:
            small_defects.append(defect)

    return big_defects, small_defects


def print_summary(
    score: float,
    big_defects: list[dict[str, object]],
    small_defects: list[dict[str, object]],
) -> None:
    """
    打印预测结果摘要
    参数:
        score: 异常分数
        big_defects: 大缺陷列表
        small_defects: 小缺陷列表
    """
    print("===================================")
    print("后端 backend: onnx")
    print(f"模型路径: {MODEL_PATH}")
    print(f"图像路径: {IMAGE_PATH}")
    print(f"图像分数 score: {score:.4f}")
    print(f"大缺陷数量: {len(big_defects)}")
    print(f"小缺陷数量: {len(small_defects)}")

    for i, defect in enumerate(big_defects):
        print(f"[大缺陷 {i}] area={defect['area']}, bbox={defect['bbox']}")

    for i, defect in enumerate(small_defects):
        print(f"[小缺陷 {i}] area={defect['area']}, bbox={defect['bbox']}")

    print("===================================")


def main() -> None:
    """
    主函数，运行图像预测管道
    """
    # 加载图像
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: Could not read image from {IMAGE_PATH}")
        return

    # 为模型预处理图像
    input_nchw = preprocess_for_model(image)

    # 运行 ONNX 推理
    score, anomaly_map = run_onnx(input_nchw)

    # 从异常图生成二值掩码
    mask = make_mask(
        anomaly_map, (image.shape[1], image.shape[0], 3)
    )  # (width, height, channels)

    # 检测和分类缺陷
    big_defects, small_defects = split_defects_by_mask(image, mask)

    # 打印结果摘要
    print_summary(score, big_defects, small_defects)


if __name__ == "__main__":
    main()
