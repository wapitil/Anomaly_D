import os
import cv2
from flask import Flask, Response

SAVE_DIR = "./Server"
TOTAL = 100

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
cap.set(cv2.CAP_PROP_FPS, 30)

app = Flask(__name__)

saved_count = 0

def gen():
    global saved_count

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 保存前100张（原图）
        if saved_count < TOTAL:
            path = os.path.join(SAVE_DIR, f"img_{saved_count:03d}.jpg")
            cv2.imwrite(path, frame)
            saved_count += 1

        # 推流用：降分辨率 + JPEG
        view = cv2.resize(frame, (1280, 720))
        ret2, jpg = cv2.imencode('.jpg', view, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret2:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')

@app.route('/video')
def video():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # 0.0.0.0 方便PC访问
    app.run(host="0.0.0.0", port=8080, threaded=True)