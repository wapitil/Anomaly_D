import logging
import os
import zipfile
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from train import train
from utils import setup_logger

# 配置LOGGER
setup_logger()
logger = logging.getLogger(__name__)

# 配置 Flask 服务器
app = Flask(__name__)


def main(save_path):
    # 解压文件
    extract_path = Path.cwd() / "Server"
    with zipfile.ZipFile(save_path, "r") as zf:
        zf.extractall(extract_path)

    save_path.unlink()  # 删除 zip 包
    logger.info("开始训练...")
    train_root = Path.cwd() / Path(save_path).stem
    logger.info(f"train_root：{train_root}")  # 这的路径是PC的路径

    # 这里方便测试，假装一下 实际的时候记得拆除
    train_root = "Server/20260423_084518"
    train(True, train_root)


@app.route("/download/<filename>", methods=["GET"])
def download(filename):
    # model_path = Path.cwd() / "Server" / "results" / filename / "model.onnx"
    # 这里方便测试，假装一下 实际的时候记得拆除
    model_path = "Server/20260423_084518/models/model.onnx"
    return send_file(model_path, as_attachment=True)


@app.route("/upload", methods=["POST"])
def upload():
    images_path = Path.cwd() / "Server"
    os.makedirs(images_path, exist_ok=True)
    file = request.files.get("file")

    if file is None:
        return jsonify({"error": "No File Exist"}), 400

    filename = Path(str(file.filename)).name
    print("filename =", filename)
    save_path = images_path / filename
    file.save(save_path)
    # 开始训练
    # main(save_path)

    return jsonify(
        {
            "message": "训练完成",
        }
    ), 200


if __name__ == "__main__":
    # 接收RDK端信息

    app.run(host="0.0.0.0", port=5000)
