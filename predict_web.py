import time
from threading import Lock, Thread

import cv2
from flask import Flask, Response

import predict

WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
JPEG_QUALITY = 80

app = Flask(__name__)
latest_jpg = None
latest_text = "等待推理结果..."
frame_lock = Lock()


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
            .bar { padding: 12px 16px; background: #222; font-size: 18px; }
            img { display: block; max-width: 100vw; max-height: calc(100vh - 50px); margin: 0 auto; }
        </style>
    </head>
    <body>
        <div class="bar">predict.py 实时结果</div>
        <img src="/video_feed">
    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(make_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    Thread(target=run_predict_loop, daemon=True).start()
    print(f"网页地址: http://{WEB_HOST}:{WEB_PORT}")
    app.run(host=WEB_HOST, port=WEB_PORT, threaded=True)


if __name__ == "__main__":
    main()
