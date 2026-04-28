import time
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import onnxruntime

# 配置路径和阈值
# IMAGE_PATH = "/app/huahong/leather/color/000.png"  # 输入图像路径
PROJECT_ROOT = Path("/app/huahong")
CURRENT_MODEL_LINK = PROJECT_ROOT / "current_model"
MODEL_NAME = "model.onnx"
MODEL_PATH = CURRENT_MODEL_LINK / MODEL_NAME  # ONNX模型路径
IMAGE_SIZE = 256  # 图像调整大小的目标尺寸
BIG_AREA_THRESH = 2000  # 将缺陷分类为“大”的阈值
MIN_AREA = 200  # 检测到的缺陷最小面积
PAD = 10
YOLO_PATCH_SIZE = 640
YOLO_NMS_THRESH = 0.7
YOLO_SCORE_THRESH = 0.25
YOLO_REG = 16
YOLO_STRIDES = [8, 16, 32]
YOLO_PAD_VALUE = 127

_YOLO_MODEL = None
_ONNX_SESSION = None
_ONNX_MODEL_REALPATH = None
_ONNX_MODEL_MTIME_NS = None
_ONNX_LOCK = Lock()


def preprocess_for_model(image_bgr: np.ndarray) -> np.ndarray:
    # 预处理图像
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))  # 调整图像尺寸
    input_float = resized.astype(np.float32) / 255.0  # 标准化为浮动数据
    input_chw = np.transpose(input_float, (2, 0, 1))  # 转换为 NCHW 格式
    return np.expand_dims(input_chw, axis=0)


def _resolve_model_path() -> Path:
    model_path = MODEL_PATH
    if model_path.exists():
        return model_path

    # TODO 后期记得删除
    # print("TODO fallback_path 记得修改")
    fallback_path = PROJECT_ROOT / "stfpm_leather_split.onnx"
    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(f"模型文件不存在: {model_path}")


def get_onnx_session() -> onnxruntime.InferenceSession:
    """按 current_model 软链接目标缓存 ONNX 会话，切换模型后自动重载。"""
    global _ONNX_SESSION, _ONNX_MODEL_REALPATH, _ONNX_MODEL_MTIME_NS

    model_path = _resolve_model_path()
    realpath = model_path.resolve()
    mtime_ns = realpath.stat().st_mtime_ns

    with _ONNX_LOCK:
        if (
            _ONNX_SESSION is not None
            and _ONNX_MODEL_REALPATH == realpath
            and _ONNX_MODEL_MTIME_NS == mtime_ns
        ):
            return _ONNX_SESSION

        print("加载 ONNX 模型:", realpath)
        _ONNX_SESSION = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],  # 使用 CPU 进行推理
        )
        _ONNX_MODEL_REALPATH = realpath
        _ONNX_MODEL_MTIME_NS = mtime_ns
        return _ONNX_SESSION


def current_onnx_model_path() -> Path:
    return _resolve_model_path().resolve()


def run_onnx(input_nchw: np.ndarray) -> tuple[float, bool, np.ndarray, np.ndarray]:
    """
    执行 ONNX 模型推理
    参数:
        input_nchw: 预处理后的图像，NCHW 格式
    返回:
        包含两个值的元组:
            - score: 异常分数
            - anomaly_map: 2D 异常图
    """
    session = get_onnx_session()
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    # 执行推理
    output_values = session.run(output_names, {input_name: input_nchw})
    outputs = dict(zip(output_names, output_values))

    # 提取异常分数、图像级判断、异常图和模型自带 mask
    score = float(np.asarray(outputs["output"]).reshape(-1)[0])
    anomaly_map = np.asarray(outputs["anomaly_map"], dtype=np.float32)
    pred_label = bool(np.asarray(outputs.get("483", [[score >= 0.5]])).reshape(-1)[0])
    model_mask = np.asarray(outputs.get("485", np.zeros_like(anomaly_map)), dtype=bool)
    return score, pred_label, anomaly_map, model_mask


def make_mask_from_model(
    model_mask: np.ndarray,
    image_shape: tuple[int, int, int],
    pred_label: bool,
) -> np.ndarray:
    h, w = image_shape[:2]
    if not pred_label:
        return np.zeros((h, w), dtype=np.uint8)

    mask_small = np.squeeze(model_mask).astype(np.uint8) * 255
    return cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)


def split_defects_by_mask(image_bgr, mask):
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )  # 8连通 = 上下左右 + 斜

    big_defects = []
    small_defects = []
    h, w = image_bgr.shape[:2]

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < MIN_AREA:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])

        x1 = max(0, x - PAD)
        y1 = max(0, y - PAD)
        x2 = min(w, x + ww + PAD)
        y2 = min(h, y + hh + PAD)

        # 每个连通域的质心
        cx = int(centroids[i][0])
        cy = int(centroids[i][1])

        defect = {
            "area": area,
            "bbox": (x1, y1, x2, y2),
            "center": (cx, cy),
        }

        if area >= BIG_AREA_THRESH:
            # print("检测到大缺陷")
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
    print(f"模型路径: {current_onnx_model_path()}")
    print(f"图像分数 score: {score:.4f}")
    print(f"大缺陷数量: {len(big_defects)}")
    print(f"小缺陷数量: {len(small_defects)}")

    for i, defect in enumerate(big_defects):
        print(f"[大缺陷 {i}] area={defect['area']}, bbox={defect['bbox']}")

    for i, defect in enumerate(small_defects):
        print(f"[小缺陷 {i}] area={defect['area']}, bbox={defect['bbox']}")

    print("===================================")


def CaptureImages():
    "从摄像头捕捉视频流"
    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("无法读取视频流")
    #         break

    images_dir = Path("/app/huahong/leather/glue")
    while True:
        for image in images_dir.iterdir():
            image_path = Path(images_dir) / image
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"无法读取图像 {image_path}")
                continue

            yield frame  # 返回当前帧
            time.sleep(1 / 5)  # 模拟摄像头帧率 30 FPS


def get_yolo_model():
    """延迟加载 YOLO，避免每帧重复加载 BPU 模型。"""
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    from yolov8_detect import Ultralytics_YOLO_Detect_Bayese_YUV420SP, load_flags

    config_path = Path(__file__).with_name("config.yaml")
    parse = load_flags(str(config_path))
    _YOLO_MODEL = Ultralytics_YOLO_Detect_Bayese_YUV420SP(
        model_path=parse["model_path"],
        classes_num=int(parse["model_nc"]),
        nms_thres=YOLO_NMS_THRESH,
        score_thres=YOLO_SCORE_THRESH,
        reg=YOLO_REG,
        strides=YOLO_STRIDES,
    )
    return _YOLO_MODEL


def run_yolo_on_image(
    image_bgr: np.ndarray,
) -> list[tuple[int, float, int, int, int, int]]:
    model = get_yolo_model()
    input_tensor = model.preprocess_yuv420sp(image_bgr)
    outputs = model.c2numpy(model.forward(input_tensor))
    return model.postProcess(outputs)


def crop_640_by_center(
    image_bgr: np.ndarray,
    center: tuple[int, int],
    patch_size: int = YOLO_PATCH_SIZE,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    以 center 为中心裁剪 patch_size x patch_size。
    如果越界，用常数补齐；返回 patch 和 patch 左上角在原图中的理论坐标。
    """
    img_h, img_w = image_bgr.shape[:2]
    cx, cy = center
    half = patch_size // 2
    patch_x1 = int(cx) - half
    patch_y1 = int(cy) - half
    patch_x2 = patch_x1 + patch_size
    patch_y2 = patch_y1 + patch_size

    src_x1 = max(0, patch_x1)
    src_y1 = max(0, patch_y1)
    src_x2 = min(img_w, patch_x2)
    src_y2 = min(img_h, patch_y2)

    patch = np.full(
        (patch_size, patch_size, image_bgr.shape[2]),
        YOLO_PAD_VALUE,
        dtype=image_bgr.dtype,
    )
    dst_x1 = src_x1 - patch_x1
    dst_y1 = src_y1 - patch_y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    patch[dst_y1:dst_y2, dst_x1:dst_x2] = image_bgr[src_y1:src_y2, src_x1:src_x2]
    return patch, (patch_x1, patch_y1)


def remap_patch_results(
    results: list[tuple[int, float, int, int, int, int]],
    patch_origin: tuple[int, int],
    image_shape: tuple[int, int, int],
) -> list[dict[str, object]]:
    origin_x, origin_y = patch_origin
    img_h, img_w = image_shape[:2]
    detections = []
    for class_id, score, x1, y1, x2, y2 in results:
        gx1 = max(0, min(img_w, int(x1 + origin_x)))
        gy1 = max(0, min(img_h, int(y1 + origin_y)))
        gx2 = max(0, min(img_w, int(x2 + origin_x)))
        gy2 = max(0, min(img_h, int(y2 + origin_y)))
        detections.append(
            {
                "class_id": int(class_id),
                "score": float(score),
                "bbox": (gx1, gy1, gx2, gy2),
            }
        )
    return detections


def RunYolo(image_bgr, big_defects, small_defects):
    """
    1. 如果有大缺陷，直接使用大 YOLO 检测整张图。
    2. 如果没有大缺陷，裁剪小缺陷为 640x640，并用小 YOLO 检测。
    3. 如果都没有则证明是正常图像。
    """
    if big_defects:
        # YOLO的一次推理就可以检测出所有大缺陷，所以不需要对每一个大缺陷进行检测
        print("正在使用大YOLO检测缺陷")
        return
        # results = run_yolo_on_image(image_bgr)
        # detections = remap_patch_results(results, (0, 0), image_bgr.shape)
        # for detection in detections:
        #     print(
        #         f"大YOLO检测结果: class_id={detection['class_id']}, "
        #         f"score={detection['score']:.3f}, bbox={detection['bbox']}"
        #     )
        # return detections

    if not small_defects:
        print("未检测到缺陷，图像正常")
        return []

    all_detections = []
    for i, defect in enumerate(small_defects):
        print("正在使用小YOLO检测缺陷")
        continue
        # patch, patch_origin = crop_640_by_center(image_bgr, defect["center"])
        # results = run_yolo_on_image(patch)
        # detections = remap_patch_results(results, patch_origin, image_bgr.shape)
        # all_detections.extend(detections)

        # print(
        #     f"小缺陷 {i}: center={defect['center']}, "
        #     f"patch_origin={patch_origin}, yolo_results={len(detections)}"
        # )
        # for detection in detections:
        #     print(
        #         f"小YOLO检测结果: class_id={detection['class_id']}, "
        #         f"score={detection['score']:.3f}, bbox={detection['bbox']}"
        #     )

    return all_detections


def main() -> None:
    """
    主函数，运行图像预测管道
    """
    # 从摄像头获取
    image_stream = CaptureImages()

    for image in image_stream:
        # 为模型预处理图像
        input_nchw = preprocess_for_model(image)

        # 运行 ONNX 推理
        score, pred_label, anomaly_map, model_mask = run_onnx(input_nchw)

        # 使用模型自带 mask，不再手写 anomaly_map 阈值划分
        mask = make_mask_from_model(model_mask, image.shape, pred_label)

        # STFPM 检测和分类缺陷
        big_defects, small_defects = split_defects_by_mask(image, mask)

        # YOLO检测
        RunYolo(image, big_defects, small_defects)
        # # 打印结果摘要
        # print_summary(score, big_defects, small_defects)


if __name__ == "__main__":
    main()
