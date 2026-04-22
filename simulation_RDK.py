import glob
import json
import os
import socket
from datetime import datetime

import cv2


def CheckNewFabric():
    "检查是否是新布"
    is_new = True
    return is_new


def CaptureImages(save_path, idx):
    "采集图像"
    # 相机采集图像

    print("1. (实际) 从摄像头开始采集")
    cap = cv2.VideoCapture(0)  # 0代表默认摄像头

    ret, frame = cap.read()
    if ret:
        image_path = os.path.join(save_path, f"img_{idx:03d}.png")
        cv2.imwrite(image_path, frame)
        print(f"图像已保存至: {image_path}")
    cap.release()


def NewFolder():
    "创建文件夹并且文件夹需要可以根据名称直接确定当前文件夹是否是最新"
    # 使用当前时间作为文件夹名称，格式为 YYYYMMDD_HHMMSS
    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        os.getcwd(), "Server", "images", folder_name, "good"
    )  # images/20260422_092624/good
    os.makedirs(save_path, exist_ok=True)
    print(f"已创建文件夹: {save_path}")
    return save_path


def UploadServer():
    print("图像采集完成，准备上传文件到服务器...")
    # TODO: 实现文件上传到服务器的逻辑
    # 例如使用 paramiko 库进行 SFTP 上传，或者使用 requests 库调用 HTTP API
    # upload_server_path = "server_upload_path" # 替换为服务器目标路径
    # for image_file in images:
    #     local_file_path = os.path.join(save_path, image_file)
    #     # 上传逻辑...
    #     print(f"上传 {local_file_path} 到服务器...")
    # print("文件上传完成")

    # TODO: 向服务器发送消息，让服务器知道RDK端已经完成了采集工作


def FakeSeed(send_path):
    """模拟通信"""
    HOST = "127.0.0.1"
    PORT = 50007

    # 1. 创建 socket 对象，使用 IPv4 和 TCP 协议
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        message = json.dumps({"path": str(send_path)})
        s.sendall(message.encode("utf-8"))


def FakeUploadServer(link_path):
    """
    is_real = False 时启用
    link_path: 链接文件存放的地址
    """
    print("[Fake] 图像采集完成，准备上传文件到服务器...")
    # 将指定文件夹内的所有文件链接至target_path

    # 注意: 一定要写绝对路径
    original_path = "/home/wapiti/Projects/Anomaly_D/Datasets/MVTecAD/leather/train/good"  # 源文件夹

    for f in os.listdir(original_path):
        source_file = os.path.join(original_path, f)  # 注意: 这个是将链接指向每一个文件
        link_target = os.path.join(link_path, f)

        # 如果链接已经存在，删除
        if os.path.islink(link_target) or os.path.exists(link_target):
            os.remove(link_target)

        os.symlink(source_file, link_target)
    num_png = len(glob.glob(link_path + "/*.png"))
    print(f"[Fake] {num_png} 张图片 正上传至{link_path}")

    # 发送上传信息
    FakeSeed(link_path)

    return link_path


if __name__ == "__main__":
    # 开始检测
    is_real = False
    is_new = CheckNewFabric()
    num_target = 1  # 这里假设是 1

    # 创建新文件夹
    save_path = NewFolder()
    if not is_real:
        FakeUploadServer(save_path)
    if is_new and is_real:
        print("检测到新布，开始采集")

        for idx in range(num_target):
            # 采集图像
            CaptureImages(save_path, idx)

        # 对文件夹进行检查是否有 num_targer 张照片
        num_png = glob.glob(save_path + "/*.png")
        if len(num_png) == num_target:
            # 上传至服务器
            UploadServer()
        else:
            raise ValueError(f"[ERROR] 当前只采集了 {len(num_png)} 张！")
