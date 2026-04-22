import json
import logging
import socket
from pathlib import Path

from train import train
from utils import setup_logger

# 配置LOGGER
setup_logger()
logger = logging.getLogger(__name__)


def Receive():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        HOST = "127.0.0.1"
        PORT = 50007
        s.bind((HOST, PORT))
        s.listen()
        print("[PC] 监听中...")

        while True:
            conn, addr = s.accept()
            with conn:
                data = conn.recv(1024)  # 接收数据（最多1024字节）
                if not data:
                    continue

                msg = json.loads(data.decode("utf-8"))
                path = msg["path"]

                print(f"[PC] 收到消息: {msg}")
                logger.info(f"receiver_path：{path}")
                return path


if __name__ == "__main__":
    # 接收RDK端信息

    is_real = True

    if not is_real:
        receiver_path = Path.cwd() / "Datasets" / "MVTecAD"
    else:
        receiver_path = Receive()
        receiver_root = Path(receiver_path).parent

    logger.info("开始训练...")
    train(is_real, receiver_root)

    # 调用训练模型
