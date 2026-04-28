import logging
import shutil
import threading
import zipfile
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from utils import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

app = Flask(__name__)

SERVER_ROOT = Path.cwd() / "Server"
UPLOAD_DIR = SERVER_ROOT / "uploads"
MODEL_NAME = "model.onnx"
MIN_TRAIN_IMAGES = 5

jobs = {}
jobs_lock = threading.Lock()


def SetJob(folder_name, status, message="", model_path=""):
    with jobs_lock:
        jobs[folder_name] = {
            "status": status,
            "message": message,
            "model_path": str(model_path),
        }


def GetJob(folder_name):
    with jobs_lock:
        return jobs.get(folder_name)


def ExtractImages(zip_path, folder_name):
    train_root = SERVER_ROOT / folder_name
    if train_root.exists():
        shutil.rmtree(train_root)
    train_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(train_root)

    zip_path.unlink(missing_ok=True)
    return train_root


def CheckTrainImages(train_root):
    good_dir = train_root / "good"
    image_paths = []
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        image_paths.extend(good_dir.glob(suffix))

    # TODO 测试完毕后，需要将此注释打开
    # if len(image_paths) < MIN_TRAIN_IMAGES:
    #     raise RuntimeError(
    #         f"训练图片数量不足: {len(image_paths)} 张，至少需要 {MIN_TRAIN_IMAGES} 张"
    #     )


def SaveModelForDownload(train_root, trained_model_path):
    """训练好的文件下载"""
    model_dir = train_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    target_path = model_dir / MODEL_NAME
    shutil.copy2(trained_model_path, target_path)
    return target_path


def TrainTask(zip_path, folder_name):
    try:
        SetJob(folder_name, "running", "正在解压图片")
        logger.info("收到训练任务: %s", folder_name)
        train_root = ExtractImages(zip_path, folder_name)
        logger.info("train_root: %s", train_root)

        # 训练函数已被注释，使用静态模型并确保其可下载
        static_model_source = Path("Datasets/MVTecAD/leather/train/models/stfpm_RDK.onnx")
  
        # 目标保存路径 (用于下载)
        download_model_dir = SERVER_ROOT / folder_name / "model"
        download_model_dir.mkdir(parents=True, exist_ok=True)
        actual_model_path_for_download = download_model_dir / MODEL_NAME
        
        # 复制静态模型到可下载位置
        shutil.copy2(static_model_source, actual_model_path_for_download)
        
        SetJob(
            folder_name,
            "ready",
            f"静态模型就绪: {static_model_source}",
            str(actual_model_path_for_download), # 保存最终可下载的路径
        )
        logger.info(
            "任务 %s: 静态模型 %s 已复制到 %s 并标记为就绪。",
            folder_name,
            static_model_source,
            actual_model_path_for_download,
        )
    except Exception as exc:
        logger.exception("训练失败: %s", folder_name)
        SetJob(folder_name, "failed", str(exc))


@app.route("/upload", methods=["POST"])
def upload():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file = request.files.get("file")

    if file is None:
        return jsonify({"error": "No File Exist"}), 400

    filename = Path(str(file.filename)).name
    folder_name = Path(filename).stem
    zip_path = UPLOAD_DIR / filename
    file.save(zip_path)

    SetJob(folder_name, "queued", "已收到图片包")
    thread = threading.Thread(
        target=TrainTask, args=(zip_path, folder_name), daemon=True
    )
    thread.start()
    logger.info("后台训练线程已启动: %s", folder_name)

    return jsonify(
        {
            "message": "已收到图片包，开始后台训练",
            "folder_name": folder_name,
        }
    ), 200


@app.route("/ready/<folder_name>", methods=["GET"])
def ready(folder_name):
    job = GetJob(folder_name)
    if job is None:
        return jsonify(
            {"ready": False, "status": "missing", "message": "任务不存在"}
        ), 404

    return jsonify(
        {
            "ready": job["status"] == "ready",
            "status": job["status"],
            "message": job["message"],
        }
    ), 200


@app.route("/download/<folder_name>", methods=["GET"])
def download(folder_name):
    job = GetJob(folder_name)
    model_path = SERVER_ROOT / folder_name / "model" / MODEL_NAME

    if job is not None and job["status"] != "ready":
        return jsonify({"error": "模型还没准备好", "status": job["status"]}), 409

    if not model_path.exists():
        return jsonify({"error": "模型文件不存在"}), 404

    return send_file(model_path, as_attachment=True, download_name=MODEL_NAME)


if __name__ == "__main__":
    SERVER_ROOT.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
