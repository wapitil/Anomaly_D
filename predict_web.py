import time
import traceback
from datetime import datetime
from threading import Lock, Thread

import cv2
from flask import Flask, Response, jsonify

import RDK_model_update
import predict

WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
JPEG_QUALITY = 80

app = Flask(__name__)
latest_jpg = None
latest_text = "等待推理结果..."
frame_lock = Lock()
update_lock = Lock()
update_state = {
    "running": False,
    "ok": None,
    "message": "未开始模型更新",
    "started_at": None,
    "finished_at": None,
    "model_path": None,
}


def set_update_state(**kwargs):
    with update_lock:
        update_state.update(kwargs)


def get_update_state():
    with update_lock:
        state = dict(update_state)

    try:
        state["model_path"] = str(predict.current_onnx_model_path())
    except Exception as exc:
        state["model_path"] = f"读取失败: {exc}"

    return state


def run_model_update():
    set_update_state(
        running=True,
        ok=None,
        message="正在采集图片并更新模型...",
        started_at=datetime.now().isoformat(timespec="seconds"),
        finished_at=None,
    )

    try:
        RDK_model_update.UpdateModel()
    except Exception as exc:
        traceback.print_exc()
        set_update_state(
            running=False,
            ok=False,
            message=f"模型更新失败: {exc}",
            finished_at=datetime.now().isoformat(timespec="seconds"),
        )
        return

    set_update_state(
        running=False,
        ok=True,
        message="模型更新完成，实时推理会自动加载新模型",
        finished_at=datetime.now().isoformat(timespec="seconds"),
    )


def start_model_update():
    with update_lock:
        if update_state["running"]:
            return False
        update_state.update(
            {
                "running": True,
                "ok": None,
                "message": "模型更新任务已启动...",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "finished_at": None,
            }
        )

    Thread(target=run_model_update, daemon=True).start()
    return True


def draw_result(image, score, big_defects, small_defects):
    show = image.copy()

    for defect in big_defects:
        x1, y1, x2, y2 = defect["bbox"]
        cv2.rectangle(show, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            show,
            "BIG",
            (x1, max(25, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    for defect in small_defects:
        x1, y1, x2, y2 = defect["bbox"]
        cv2.rectangle(show, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            show,
            "SMALL",
            (x1, max(25, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    text = f"score={score:.4f}  big={len(big_defects)}  small={len(small_defects)}"
    cv2.rectangle(show, (0, 0), (show.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(show, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return show, text


def run_predict_loop():
    global latest_jpg, latest_text

    image_stream = predict.CaptureImages()
    for image in image_stream:
        # 模型预处理
        input_nchw = predict.preprocess_for_model(image)

        score, pred_label, anomaly_map, model_mask = predict.run_onnx(input_nchw)
        mask = predict.make_mask_from_model(model_mask, image.shape, pred_label)
        big_defects, small_defects = predict.split_defects_by_mask(image, mask)

        predict.RunYolo(image, big_defects, small_defects)

        show, text = draw_result(image, score, big_defects, small_defects)
        ok, jpg = cv2.imencode(".jpg", show, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            continue

        with frame_lock:
            latest_jpg = jpg.tobytes()
            latest_text = text


def make_stream():
    while True:
        with frame_lock:
            frame = latest_jpg

        if frame is None:
            time.sleep(0.1)
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


@app.route("/")
def index():
    return """
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Predict Web</title>
        <style>
            body { margin: 0; background: #111; color: #eee; font-family: Arial, sans-serif; }
            .bar {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 12px 16px;
                background: #222;
                font-size: 18px;
            }
            .title { font-weight: 700; }
            .status {
                flex: 1;
                min-width: 0;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                color: #bbb;
                font-size: 14px;
            }
            button {
                border: 0;
                border-radius: 6px;
                background: #18a058;
                color: white;
                cursor: pointer;
                font-size: 14px;
                padding: 8px 12px;
            }
            button:disabled { background: #555; cursor: wait; }
            img { display: block; max-width: 100vw; max-height: calc(100vh - 58px); margin: 0 auto; }
        </style>
    </head>
    <body>
        <div class="bar">
            <div class="title">predict.py 实时结果</div>
            <div class="status" id="status">正在读取模型状态...</div>
            <button id="updateBtn" type="button">更新模型</button>
        </div>
        <img src="/video_feed">
        <script>
            const statusEl = document.getElementById("status");
            const updateBtn = document.getElementById("updateBtn");

            async function refreshStatus() {
                const response = await fetch("/model_update/status");
                const data = await response.json();
                updateBtn.disabled = data.running;
                statusEl.textContent = `${data.message} | 当前模型: ${data.model_path || "未知"}`;
            }

            updateBtn.addEventListener("click", async () => {
                updateBtn.disabled = true;
                const response = await fetch("/model_update", { method: "POST" });
                const data = await response.json();
                statusEl.textContent = data.message;
                refreshStatus();
            });

            refreshStatus();
            setInterval(refreshStatus, 2000);
        </script>
    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(make_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/model_update", methods=["POST"])
def model_update():
    started = start_model_update()
    status_code = 202 if started else 409
    state = get_update_state()
    return jsonify(state), status_code


@app.route("/model_update/status")
def model_update_status():
    return jsonify(get_update_state())

def run_camara():
    """ 实时推流 """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 设置宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 设置高度
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频流")
            break
        yield frame  # 返回当前帧
        time.sleep(1 / 30)  # 设置帧率为30 FPS
        
def main():
    # Thread(target=run_predict_loop, daemon=True).start() # 整体流程
    Thread(target=run_camara, daemon=True).start() # 实时推流
    print(f"网页地址: http://{WEB_HOST}:{WEB_PORT}")
    app.run(host=WEB_HOST, port=WEB_PORT, threaded=True)


if __name__ == "__main__":
    main()
