from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread

import cv2
from flask import Flask, Response, jsonify

import predict

WEB_HOST = "0.0.0.0"
WEB_PORT = 5001
NORMAL_DIR = Path("pipeline/normal_images")
CAPTURE_ROOT = Path("Server")

app = Flask(__name__)
frame_lock = Lock()
latest_jpg = None
update_info = {"status": "idle", "message": "检测中"}


def run_predict_loop() -> None:
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
        <title>BPU Anomaly</title>
        <style>
            body { margin: 0; background: #111; color: #eee; font-family: Arial, sans-serif; }
            .bar { display: flex; gap: 24px; align-items: center; padding: 12px 16px; background: #222; font-size: 18px; }
            button { font-size: 16px; padding: 6px 12px; }
            img { display: block; max-width: 100vw; max-height: calc(100vh - 54px); margin: 0 auto; }
        </style>
    </head>
    <body>
        <div class="bar">
            <span>BPU 实时异常检测</span>
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
    return jsonify(update_info)


@app.route("/new_fabric", methods=["POST"])
def new_fabric():
    if update_info["status"] not in {"idle", "done", "error"}:
        return jsonify({"status": update_info["status"], "message": "当前正在更新中，请不要重复点击"})

    update_info["status"] = "capturing"
    update_info["message"] = "收到新布信号，开始采集正常图"
    Thread(target=update_stats_task, daemon=True).start()
    return jsonify(update_info)


def capture_normal_images() -> Path:
    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_dir = CAPTURE_ROOT / folder_name / "good"
    image_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("摄像头打开失败")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        target = 20
        for i in range(target):
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            cv2.imwrite(str(image_dir / f"img_{i:03d}.png"), frame)
            time.sleep(0.2)
    finally:
        cap.release()

    return image_dir


def replace_normal_images(image_dir: Path) -> None:
    NORMAL_DIR.mkdir(parents=True, exist_ok=True)
    for old_image in NORMAL_DIR.glob("*.png"):
        old_image.unlink()
    for image_path in sorted(image_dir.glob("*.png")):
        target_path = NORMAL_DIR / image_path.name
        target_path.write_bytes(image_path.read_bytes())


def update_stats_task() -> None:
    try:
        update_info["status"] = "capturing"
        update_info["message"] = "正在采集正常图"
        image_dir = capture_normal_images()

        update_info["status"] = "fitting"
        update_info["message"] = "正在用新正常图更新阈值"
        replace_normal_images(image_dir)
        predict.fit_stats(NORMAL_DIR)

        update_info["status"] = "done"
        update_info["message"] = f"阈值更新完成: {image_dir}"
    except Exception as exc:
        update_info["status"] = "error"
        update_info["message"] = f"更新失败: {exc}"


def main() -> None:
    Thread(target=run_predict_loop, daemon=True).start()
    print(f"网页地址: http://{WEB_HOST}:{WEB_PORT}")
    app.run(host=WEB_HOST, port=WEB_PORT, threaded=True)


if __name__ == "__main__":
    main()
