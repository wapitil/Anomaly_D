import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
from PIL import Image

MODEL_PATH = Path("model.onnx")
IMAGE_SIZE = 256

session = None


def get_model_path():
    current_model = Path("current_model") / "model.onnx"
    if current_model.exists():
        return current_model
    return MODEL_PATH


def get_session():
    global session  # 避免重复加载模型环境

    if session is None:
        model_path = get_model_path()
        print("加载 PaDiM 模型:", model_path)
        session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

    return session


def reload_model():
    """ 让下一次推理重新加载新的模型 """
    global session
    session = None
    print("模型会在下一次推理时重新加载")


def preprocess_for_model(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)

    data = np.asarray(image, dtype=np.float32) / 255.0
    data = np.transpose(data, (2, 0, 1))
    data = np.expand_dims(data, axis=0)
    return data


def run_onnx(input_data):
    sess = get_session()
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: input_data})

    score = float(np.squeeze(outputs[0]))  # 异常分数
    anomaly_map = outputs[1]  # 异常图
    pred_label = bool(np.squeeze(outputs[2]))  # 是否异常
    pred_mask = outputs[3]  # 异常mask

    return score, anomaly_map, pred_label, pred_mask


def make_mask_from_model(pred_mask, pred_label):
    "对pred_mask进行处理，得到的mask更干净"
    if not pred_label:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    mask = np.squeeze(pred_mask).astype(np.uint8) * 255
    return mask


def draw_result(image, anomaly_map, mask, score, pred_label):
    # 原图尺寸
    h, w = image.shape[:2]

    # 如果模型判断正常，直接返回原图
    if not pred_label:
        return image

    # mask 转成 2D
    mask_2d = np.squeeze(mask)

    # resize 到原图尺寸
    mask_2d = cv2.resize(
        mask_2d.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )

    # 转成 0/255
    binary_mask = (mask_2d > 0).astype(np.uint8) * 255

    # 找轮廓
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # 复制原图，避免改原始 image
    result = image.copy()

    # 画轮廓：红色，线宽 2
    cv2.drawContours(
        result,
        contours,
        -1,
        (0, 0, 255),
        2,
    )

    return result


def predict_image(image):
    input_data = preprocess_for_model(image)
    score, anomaly_map, pred_label, pred_mask = run_onnx(input_data)
    mask = make_mask_from_model(pred_mask, pred_label)
    result = draw_result(image, anomaly_map, mask, score, pred_label)

    return {
        "score": score,
        "pred_label": pred_label,
        "mask": mask,
        "mask_pixels": int(np.count_nonzero(mask)),
        "result": result,
    }


def image_stream():
    """需要将这个修改成从摄像头捕捉画面"""
    image_dir = Path("leather/test/color")

    while True:
        for image_path in sorted(image_dir.glob("*.png")):
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            yield image
            time.sleep(0.2)  # 1 / 5 帧率


def main():
    for image in image_stream():
        predict_image(image)


if __name__ == "__main__":
    main()
