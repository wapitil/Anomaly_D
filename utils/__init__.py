import os
import cv2
import numpy as np
import yaml
from myLogger import logger

def is_usb_camera(device):
    try:
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            return False
        cap.release()
        return True
    except Exception:
        return False

def find_first_usb_camera():
    video_devices = [os.path.join('/dev', dev) for dev in os.listdir('/dev') if dev.startswith('video')]
    for dev in video_devices:
        if is_usb_camera(dev):
            return dev
    return None

# class_names = [
#     'WEISONGSI',
#     'DUANMU',
#     'DUKONG',
#     'WANGBA',
#     'QITA'
# ]
class_names = [
    '纬松丝',
    '断目',
    '堵孔',
    '网疤',
    '其他'
]
# coco_names = [
#     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
#     "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
#     "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
#     "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
#     "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
#     "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", 
#     "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
#     "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
#     ]

rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),(49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),(147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]


def class_id2name(ids):
    tmp = []
    if isinstance(ids, int):
        tmp.append(ids)
    elif isinstance(ids, list):
        tmp = [i for i in ids if isinstance(i, int)]
    else:
        tmp = []
        for i in ids:
            tmp.append(class_names[i])
    return ','.join(tmp)


def draw_detection(img: np.array, 
                   bbox,#: tuple[int, int, int, int],
                   score: float, 
                   class_id: int) -> None:
    """
    Draws a detection bounding box and label on the image.

    Parameters:
        img (np.array): The input image.
        bbox (tuple[int, int, int, int]): A tuple containing the bounding box coordinates (x1, y1, x2, y2).
        score (float): The detection score of the object.
        class_id (int): The class ID of the detected object.
    """
    x1, y1, x2, y2 = bbox
    color = rdk_colors[class_id%20]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{class_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)




def load_flags(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) # 安全加载
    
parse = load_flags('config.yaml')
