import json
import re
import shutil
import time
from datetime import datetime
from html import escape
from pathlib import Path
from threading import Lock, Thread

import cv2
from flask import Flask, Response, abort, jsonify, request, send_file

WEB_HOST = "0.0.0.0"
WEB_PORT = 5002

app = Flask(__name__)
frame_lock = Lock()
latest_jpg = None

# 拍照相关
BASE_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = BASE_DIR / "captured_batches"
COLLECTED_DIR = BASE_DIR / "collected_data"
CAPTURE_DIR.mkdir(exist_ok=True)
COLLECTED_DIR.mkdir(exist_ok=True)
capture_request = {"pending": False}

THUMB_DIRNAME = ".thumbs"
THUMB_MAX_SIZE = 360
BATCH_RE = re.compile(r"^\d{8}_\d{6}$")
IMAGE_RE = re.compile(r"^img_\d{3}\.png$")

# 用于 Web 显示的缩放尺寸（缩小以提升实时性）
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
DISPLAY_FPS = 5
CAMERA_FPS = 5
PREVIEW_CAMERA_WIDTH = 640
PREVIEW_CAMERA_HEIGHT = 480
CAPTURE_CAMERA_WIDTH = 2560
CAPTURE_CAMERA_HEIGHT = 1920
CAPTURE_COUNT = 10
CAPTURE_INTERVAL_SEC = 0.5


# 拍照缓冲队列：capture_loop 往里面放帧，do_capture 从里面取
capture_frame_queue = []
capture_frame_lock = Lock()
capture_in_progress = False
last_capture_result = None


def _set_camera_size(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def capture_loop():
    """只采集摄像头画面，不做任何预测"""
    global latest_jpg

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("摄像头打开失败")

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    _set_camera_size(cap, PREVIEW_CAMERA_WIDTH, PREVIEW_CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    try:
        last_read_at = 0
        last_display_at = 0
        next_capture_frame_at = 0
        capture_high_res = False
        while True:
            now = time.monotonic()
            wait_time = (1 / CAMERA_FPS) - (now - last_read_at)
            if wait_time > 0:
                time.sleep(wait_time)
            last_read_at = time.monotonic()

            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # 检查是否有拍照请求（用原始分辨率保存）
            if capture_request["pending"]:
                capture_request["pending"] = False
                global capture_in_progress
                with capture_frame_lock:
                    capture_frame_queue.clear()
                _set_camera_size(cap, CAPTURE_CAMERA_WIDTH, CAPTURE_CAMERA_HEIGHT)
                next_capture_frame_at = 0
                capture_high_res = True
                capture_in_progress = True
                Thread(target=do_capture, daemon=True).start()
                continue

            # 如果正在拍照，把帧加入队列
            if capture_in_progress:
                now = time.monotonic()
                if now >= next_capture_frame_at:
                    with capture_frame_lock:
                        if len(capture_frame_queue) < CAPTURE_COUNT:
                            capture_frame_queue.append(frame.copy())
                    next_capture_frame_at = now + CAPTURE_INTERVAL_SEC
            elif capture_high_res:
                _set_camera_size(cap, PREVIEW_CAMERA_WIDTH, PREVIEW_CAMERA_HEIGHT)
                capture_high_res = False

            now = time.monotonic()
            if now - last_display_at < 1 / DISPLAY_FPS:
                if not capture_in_progress:
                    time.sleep(0.01)
                continue
            last_display_at = now

            # 缩小图像再编码为 JPEG，大幅提升实时性
            display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                continue

            with frame_lock:
                latest_jpg = jpg.tobytes()
    finally:
        cap.release()


def do_capture():
    """按固定时间间隔从队列中取帧保存为 PNG（原始分辨率）"""
    global capture_in_progress, last_capture_result

    batch_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = CAPTURE_DIR / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    max_wait = CAPTURE_COUNT * CAPTURE_INTERVAL_SEC + 5
    for _ in range(int(max_wait / 0.1)):
        frame = None
        with capture_frame_lock:
            if capture_frame_queue:
                frame = capture_frame_queue.pop(0)

        if frame is not None:
            cv2.imwrite(str(batch_dir / f"img_{saved:03d}.png"), frame)
            saved += 1
            if saved >= CAPTURE_COUNT:
                break

        time.sleep(0.1)

    capture_in_progress = False

    # 写入批次信息文件，标记为"待筛选"
    info = {
        "batch": batch_name,
        "status": "pending",
        "total": saved,
        "capture_count": CAPTURE_COUNT,
        "capture_interval_sec": CAPTURE_INTERVAL_SEC,
    }
    with open(batch_dir / "info.json", "w") as f:
        json.dump(info, f)

    last_capture_result = {
        "batch": batch_name,
        "saved": saved,
        "interval_sec": CAPTURE_INTERVAL_SEC,
    }
    print(f"📸 拍照完成: {batch_dir} ({saved} 张)")


def make_stream():
    while True:
        with frame_lock:
            frame = latest_jpg

        if frame is None:
            time.sleep(0.1)
            continue

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.03)


def _safe_batch_dir(batch_name):
    if not batch_name or not BATCH_RE.match(batch_name):
        abort(400)

    batch_dir = (CAPTURE_DIR / batch_name).resolve()
    if CAPTURE_DIR.resolve() not in batch_dir.parents:
        abort(400)
    return batch_dir


def _safe_photo_path(batch_name, filename):
    if not filename or not IMAGE_RE.match(filename):
        abort(400)

    photo_path = (_safe_batch_dir(batch_name) / filename).resolve()
    if CAPTURE_DIR.resolve() not in photo_path.parents:
        abort(400)
    return photo_path


def _thumbnail_path(photo_path):
    thumb_dir = photo_path.parent / THUMB_DIRNAME
    thumb_dir.mkdir(exist_ok=True)
    return thumb_dir / f"{photo_path.stem}.jpg"


def _ensure_thumbnail(photo_path):
    thumb_path = _thumbnail_path(photo_path)
    if thumb_path.exists() and thumb_path.stat().st_mtime >= photo_path.stat().st_mtime:
        return thumb_path

    image = cv2.imread(str(photo_path))
    if image is None:
        abort(404)

    height, width = image.shape[:2]
    scale = min(THUMB_MAX_SIZE / max(width, height), 1)
    if scale < 1:
        new_size = (int(width * scale), int(height * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    ok = cv2.imwrite(str(thumb_path), image, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        abort(500)
    return thumb_path


@app.route("/")
def index():
    return """
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>摄像头实时画面</title>
        <style>
            body { margin: 0; background: #111; color: #eee; font-family: Arial, sans-serif; }
            .bar {
                display: flex;
                gap: 24px;
                padding: 12px 16px;
                background: #222;
                font-size: 18px;
                align-items: center;
            }
            .bar span { color: #0f0; }
            .bar button {
                padding: 8px 20px;
                font-size: 16px;
                background: #0a0;
                color: #fff;
                border: none;
                border-radius: 6px;
                cursor: pointer;
            }
            .bar button:disabled {
                background: #555;
                cursor: not-allowed;
            }
            .bar button:hover:not(:disabled) { background: #0c0; }
            img { display: block; max-width: 100vw; max-height: calc(100vh - 50px); margin: 0 auto; }
        </style>
    </head>
    <body>
        <div class="bar">
            <span>📷 摄像头实时画面（无检测）</span>
            <button id="captureBtn" onclick="startCapture()">📸 拍照 10 张 / 5 秒</button>
            <span id="status"></span>
        </div>
        <img src="/video_feed">
        <script>
            async function startCapture() {
                const btn = document.getElementById("captureBtn");
                const status = document.getElementById("status");
                btn.disabled = true;
                status.textContent = "⏳ 拍照中，请稍候...";

                const resp = await fetch("/capture", { method: "POST" });
                const data = await resp.json();

                if (data.success) {
                    await waitCaptureDone(status);
                } else {
                    status.textContent = "❌ 拍照失败: " + data.error;
                    btn.disabled = false;
                }
            }

            async function waitCaptureDone(status) {
                for (let i = 0; i < 80; i++) {
                    await new Promise(resolve => setTimeout(resolve, 250));
                    const resp = await fetch("/capture_status");
                    const data = await resp.json();
                    if (!data.pending && !data.in_progress && data.last) {
                        status.textContent = `✅ 拍照完成，已保存 ${data.last.saved} 张，可前往照片管理查看`;
                        document.getElementById("captureBtn").disabled = false;
                        return;
                    }
                }
                status.textContent = "⚠️ 拍照仍在进行，请稍后前往照片管理查看";
                document.getElementById("captureBtn").disabled = false;
            }
        </script>
        <div class="bar" style="margin-top: 0; border-top: 1px solid #333;">
            <a href="/manager" style="color: #0af; text-decoration: none; font-size: 16px;">🖼️ 前往照片管理 →</a>
        </div>
    </body>
    </html>
    """


@app.route("/capture", methods=["POST"])
def trigger_capture():
    """触发拍照"""
    global last_capture_result
    if capture_request["pending"] or capture_in_progress:
        return jsonify({"success": False, "error": "已有拍照任务正在进行"})
    last_capture_result = None
    capture_request["pending"] = True
    return jsonify({"success": True})


@app.route("/capture_status")
def capture_status():
    return jsonify({
        "pending": capture_request["pending"],
        "in_progress": capture_in_progress,
        "last": last_capture_result,
    })


@app.route("/video_feed")
def video_feed():
    return Response(make_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/manager")
def photo_manager():
    """照片管理页面"""
    batches = sorted((p for p in CAPTURE_DIR.iterdir() if p.is_dir()), reverse=True)
    batch_list = []
    for batch_dir in batches:
        if batch_dir.name == THUMB_DIRNAME:
            continue
        info_path = batch_dir / "info.json"
        if not info_path.exists():
            continue
        try:
            with open(info_path) as f:
                info = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        images = sorted(batch_dir.glob("img_*.png"))
        batch_list.append({
            "batch": info.get("batch", batch_dir.name),
            "status": info.get("status", "pending"),
            "images": [img.name for img in images],
            "total": len(images),
        })

    return f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>照片管理</title>
        <style>
            body {{ margin: 20px; background: #111; color: #eee; font-family: Arial, sans-serif; }}
            h1 {{ color: #0f0; }}
            .batch {{
                background: #222;
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 20px;
            }}
            .batch h2 {{ margin: 0 0 12px 0; font-size: 16px; color: #aaa; }}
            .batch-tools {{
                display: flex;
                gap: 10px;
                margin-bottom: 12px;
                flex-wrap: wrap;
            }}
            .images {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
                gap: 12px;
            }}
            .image-card {{
                background: #333;
                border-radius: 6px;
                padding: 8px;
                text-align: center;
            }}
            .image-card img {{
                width: 100%;
                aspect-ratio: 4 / 3;
                object-fit: contain;
                background: #111;
                border-radius: 4px;
            }}
            .image-card label {{
                display: block;
                margin-top: 6px;
                cursor: pointer;
            }}
            .image-card input[type="checkbox"] {{
                transform: scale(1.3);
                margin-right: 6px;
            }}
            .actions {{
                margin-top: 12px;
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
            }}
            .actions button, .batch-tools button {{
                padding: 8px 24px;
                font-size: 15px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
            }}
            .btn-keep {{
                background: #0a0;
                color: #fff;
            }}
            .btn-keep:hover {{ background: #0c0; }}
            .btn-delete {{
                background: #a00;
                color: #fff;
            }}
            .btn-delete:hover {{ background: #c00; }}
            .btn-back {{
                background: #555;
                color: #fff;
                text-decoration: none;
                padding: 8px 24px;
                border-radius: 6px;
                display: inline-block;
            }}
            .btn-back:hover {{ background: #777; }}
            .status-badge {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 12px;
                margin-left: 8px;
            }}
            .status-pending {{ background: #a80; }}
            .status-kept {{ background: #080; }}
            .status-deleted {{ background: #800; }}
            .empty {{ color: #666; text-align: center; padding: 40px; }}
        </style>
    </head>
    <body>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
            <h1>🖼️ 照片管理</h1>
            <a href="/" class="btn-back">← 返回摄像头</a>
        </div>
        {_render_batches(batch_list)}
        <script>
            async function confirmKeep(batchName) {{
                const checkboxes = document.querySelectorAll(`input[name="${{batchName}}"]:checked`);
                const keepList = Array.from(checkboxes).map(cb => cb.value);

                if (keepList.length === 0) {{
                    alert("请至少选择一张照片");
                    return;
                }}

                if (!confirm(`确定保留 ${{keepList.length}} 张照片？未选择的将被删除。`)) {{
                    return;
                }}

                const resp = await fetch("/confirm", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{ batch: batchName, keep: keepList }})
                }});
                const data = await resp.json();
                if (data.success) {{
                    location.reload();
                }} else {{
                    alert("操作失败: " + data.error);
                }}
            }}

            async function deleteBatch(batchName) {{
                if (!confirm("确定删除整个批次？")) return;
                const resp = await fetch("/delete_batch", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{ batch: batchName }})
                }});
                const data = await resp.json();
                if (data.success) {{
                    location.reload();
                }} else {{
                    alert("删除失败: " + data.error);
                }}
            }}

            function setBatchChecked(batchName, checked) {{
                document.querySelectorAll(`input[name="${{batchName}}"]:not(:disabled)`).forEach(cb => {{
                    cb.checked = checked;
                }});
            }}
        </script>
    </body>
    </html>
    """


def _render_batches(batch_list):
    if not batch_list:
        return '<div class="empty">暂无拍照记录，请先拍照</div>'

    html = ""
    for batch in batch_list:
        batch_name = escape(batch["batch"])
        status_map = {
            "pending": '<span class="status-badge status-pending">待筛选</span>',
            "kept": '<span class="status-badge status-kept">已保留</span>',
            "deleted": '<span class="status-badge status-deleted">已删除</span>',
        }
        badge = status_map.get(batch["status"], "")

        html += f'<div class="batch">'
        html += f'<h2>📁 {batch_name}（{batch["total"]} 张） {badge}</h2>'
        if batch["status"] == "pending":
            batch_arg = json.dumps(batch["batch"])
            html += f'''
            <div class="batch-tools">
                <button type="button" onclick='setBatchChecked({batch_arg}, true)'>全选</button>
                <button type="button" onclick='setBatchChecked({batch_arg}, false)'>清空</button>
            </div>
            '''
        html += f'<div class="images">'

        for img_name in batch["images"]:
            img_path = f"/photo/{batch['batch']}/{img_name}"
            thumb_path = f"/photo_thumb/{batch['batch']}/{img_name}"
            img_label = escape(img_name)
            html += f'''
            <div class="image-card">
                <a href="{img_path}" target="_blank">
                    <img src="{thumb_path}" loading="lazy" alt="{img_label}">
                </a>
                <label>
                    <input type="checkbox" name="{batch['batch']}" value="{img_name}"
                           {"disabled" if batch["status"] != "pending" else ""}
                           {"checked" if batch["status"] == "kept" else ""}>
                    {img_label}
                </label>
            </div>
            '''

        html += '</div>'

        if batch["status"] == "pending":
            batch_arg = json.dumps(batch["batch"])
            html += f'''
            <div class="actions">
                <button class="btn-keep" onclick='confirmKeep({batch_arg})'>✅ 确认保留</button>
                <button class="btn-delete" onclick='deleteBatch({batch_arg})'>🗑️ 删除批次</button>
            </div>
            '''

        html += '</div>'

    return html


@app.route("/photo/<batch>/<filename>")
def serve_photo(batch, filename):
    """提供照片原图"""
    photo_path = _safe_photo_path(batch, filename)
    if not photo_path.exists():
        abort(404)
    return send_file(photo_path, mimetype="image/png", conditional=True, max_age=3600)


@app.route("/photo_thumb/<batch>/<filename>")
def serve_photo_thumb(batch, filename):
    """提供照片缩略图，避免管理页一次性加载大 PNG。"""
    photo_path = _safe_photo_path(batch, filename)
    if not photo_path.exists():
        abort(404)
    thumb_path = _ensure_thumbnail(photo_path)
    return send_file(thumb_path, mimetype="image/jpeg", conditional=True, max_age=86400)


@app.route("/confirm", methods=["POST"])
def confirm_keep():
    """确认保留选中的照片"""
    data = request.get_json()
    batch_name = data.get("batch")
    keep_list = data.get("keep", [])

    batch_dir = _safe_batch_dir(batch_name)
    if not batch_dir.exists():
        return jsonify({"success": False, "error": "批次不存在"})

    # 移动保留的照片到 collected_data
    collected_dir = COLLECTED_DIR / batch_name
    collected_dir.mkdir(parents=True, exist_ok=True)

    kept_count = 0
    for img_name in keep_list:
        if not IMAGE_RE.match(img_name):
            continue
        src = batch_dir / img_name
        if src.exists():
            dst = collected_dir / img_name
            src.rename(dst)
            kept_count += 1

    # 删除批次目录中剩余的文件
    shutil.rmtree(batch_dir)

    # 写入 collected_data 的 info
    info = {"batch": batch_name, "status": "kept", "kept": kept_count}
    with open(collected_dir / "info.json", "w") as f:
        json.dump(info, f)

    print(f"✅ 保留 {kept_count} 张照片到 {collected_dir}")
    return jsonify({"success": True, "kept": kept_count})


@app.route("/delete_batch", methods=["POST"])
def delete_batch():
    """删除整个批次"""
    data = request.get_json()
    batch_name = data.get("batch")

    batch_dir = _safe_batch_dir(batch_name)
    if not batch_dir.exists():
        return jsonify({"success": False, "error": "批次不存在"})

    shutil.rmtree(batch_dir)

    print(f"🗑️ 删除批次: {batch_name}")
    return jsonify({"success": True})


def main():
    Thread(target=capture_loop, daemon=True).start()
    print(f"🌐 摄像头纯画面地址: http://{WEB_HOST}:{WEB_PORT}")
    print(f"🖼️  照片管理地址: http://{WEB_HOST}:{WEB_PORT}/manager")
    app.run(host=WEB_HOST, port=WEB_PORT, threaded=True)


if __name__ == "__main__":
    main()
