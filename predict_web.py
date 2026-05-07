import time
import zipfile
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from urllib import response

import cv2
import requests
from flask import Flask, Response

import predict

WEB_HOST = "0.0.0.0"
WEB_PORT = 5001

app = Flask(__name__)
frame_lock = Lock()
latest_jpg = None
update_info = {
    "status": "idle",
    "message": "检测中",
}


def run_predict_loop():
    global latest_jpg

    for image in predict.image_stream():
        info = predict.predict_image(image)
        ok, jpg = cv2.imencode(".jpg", info["result"])
        if not ok:
            continue

        with frame_lock:
            latest_jpg = jpg.tobytes()


def make_stream():
    while True:
        with frame_lock:
            frame = latest_jpg

        if frame is None:
            time.sleep(0.1)
            continue

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.03)


@app.route("/")
def index():
    return """
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>PaDiM</title>
        <style>
            body { margin: 0; background: #111; color: #eee; font-family: Arial, sans-serif; }
            .bar {
                display: flex;
                gap: 24px;
                padding: 12px 16px;
                background: #222;
                font-size: 18px;
            }
            img { display: block; max-width: 100vw; max-height: calc(100vh - 50px); margin: 0 auto; }
        </style>
    </head>
    <body>
        <div class="bar">
            <span>PaDiM 实时检测</span>
            <span id="status">正在读取状态...</span>
            <button id="newFabricBtn" type="button">新布来了</button>
        </div>
        <img src="/video_feed">
        <script>
            async function refreshStatus() {
                const response = await fetch("/update_status");
                const data = await response.json();
                document.getElementById("status").textContent = data.message;
            }

            async function newFabric() {
                await fetch("/new_fabric", { method: "POST" });
                refreshStatus();
            }

            document.getElementById("newFabricBtn").addEventListener("click", newFabric);

            refreshStatus();
            setInterval(refreshStatus, 1000);
        </script>
    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(make_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/update_status")
def get_update_status():
    """显示流程运行状态"""
    return update_info


@app.route("/new_fabric", methods=["POST"])
def new_fabric():
    if update_info["status"] not in ["idle", "done", "error"]:
        return {
            "status": update_info["status"],
            "message": "当前正在更新中，请不要重复点击",
        }

    update_info["status"] = "capturing"
    update_info["message"] = "收到新布信号，开始更新"

    Thread(target=update_model_task, daemon=True).start()

    return update_info


def capture_images():
    """新布来了，进行采集"""
    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_dir = Path("Server") / folder_name / "good"
    image_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("摄像头打开失败")

    target = 20  # 收集 20 张照片
    for i in range(target):
        ret, frame = cap.read()
        image_path = image_dir / f"img_{i:03d}.png"
        cv2.imwrite(str(image_path), frame)

        time.sleep(1 / 5)

    cap.release()
    return folder_name, image_dir


def zip_images(folder_name):
    """
    对采集到的文件进行压缩
    zip 里的结构为:
        good/img_000.png
        good/img_001.png
    """
    version_dir = Path("Server") / folder_name
    zip_path = version_dir / f"{folder_name}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        image_dir = version_dir / "good"

        for image_path in image_dir.glob("*.png"):
            zf.write(image_path, image_path.relative_to(version_dir))

    return zip_path


def upload_zipfile(zip_path):
    """将打包的 zip 文件上传至服务器"""
    url = "http://100.110.49.18:5000/upload"
    with open(zip_path, "rb") as f:
        files = {"file": (zip_path.name, f, "application/zip")}
        response = requests.post(url, files=files, timeout=60)

    # print("服务器状态码:", response.status_code)
    # print("服务器返回:", response.text)
    response.raise_for_status()
    return response.json()


def check_model(folder_name):
    """
    检查模型是否训练好
    最多等 300s 每 5s 询问一次
    """
    max_wait = 60
    url = f"http://100.110.49.18:5000/ready/{folder_name}"
    for i in range(max_wait):
        update_info["status"] = "training"
        update_info["message"] = f"服务器训练中，第{i + 1}次等待"
        response = requests.get(url, timeout=10)

        response.raise_for_status()
        data = response.json()
        ready = data.get("ready", False)

        if ready:
            update_info["message"] = "训练完成"
            return

        time.sleep(5)
    raise RuntimeError("等待超时")


def download_model(folder_name):
    """避免下载失败，先报错tmp 成功后再替换成 model.onnx"""
    url = f"http://100.110.49.18:5000/download/{folder_name}"

    response = requests.get(url, timeout=60)

    print("下载状态码:", response.status_code)
    print("下载返回长度:", len(response.content))

    response.raise_for_status()

    model_dir = Path("Server") / folder_name / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = model_dir / "model.onnx.tmp"
    model_path = model_dir / "model.onnx"

    with open(tmp_path, "wb") as f:
        f.write(response.content)

    if tmp_path.stat().st_size == 0:
        raise RuntimeError("下载到的模型文件为空")

    tmp_path.replace(model_path)

    return model_path


def switch_current_model(model_path):
    """model_path Server/xxxx/model/model.onnx"""
    model_dir = model_path.parent

    current_model = Path("current_model")
    tmp_link = Path("current_model.tmp")

    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()

    if current_model.exists() or current_model.is_symlink():
        current_model.unlink()

    tmp_link.symlink_to(model_dir, target_is_directory=True)
    tmp_link.rename(current_model)

    predict.reload_model()

    print("当前模型已切换到:", model_dir)


def update_model_task():
    """模拟更新"""
    try:
        update_info["status"] = "capturing"
        update_info["message"] = "正在采集新布图片"

        folder_name, image_dir = capture_images()

        update_info["message"] = "正在打包图片"
        zip_path = zip_images(folder_name)

        update_info["status"] = "uploading"
        update_info["message"] = "正在上传图片"
        server_reply = upload_zipfile(zip_path)

        update_info["status"] = "training"
        update_info["message"] = "图片上传完成，等待服务器训练"
        folder_name = server_reply["folder_name"]

        check_model(folder_name)

        update_info["status"] = "downloading"
        update_info["message"] = "正在下载新模型"
        model_path = download_model(folder_name)

        update_info["message"] = "正在切换当前模型"
        switch_current_model(model_path)

        update_info["status"] = "done"
        update_info["message"] = f"模型更新完成，已切换到: {model_path}"

    except Exception as e:
        update_info["status"] = "error"
        update_info["message"] = f"更新失败: {e}"


def main():
    Thread(target=run_predict_loop, daemon=True).start()
    print(f"网页地址: http://{WEB_HOST}:{WEB_PORT}")
    app.run(host=WEB_HOST, port=WEB_PORT, threaded=True)


if __name__ == "__main__":
    main()
