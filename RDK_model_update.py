import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import requests

PC_URL = "http://100.110.49.18:5000"
PROJECT_ROOT = Path("/app/huahong")
SERVER_ROOT = PROJECT_ROOT / "Server"
CURRENT_MODEL_LINK = PROJECT_ROOT / "current_model"
MODEL_NAME = "model.onnx"
TARGET_IMAGE_COUNT = 1


def CheckNewFabric():
    "检查是否是新布"
    return True


def NewFolder():
    "为本次新布创建一个独立版本文件夹"
    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_root = SERVER_ROOT / folder_name
    image_dir = version_root / "good"
    model_dir = version_root / "model"
    image_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    print("已创建版本文件夹:", version_root)
    return folder_name, version_root, image_dir, model_dir


def CaptureImages(image_dir, idx):
    "采集图片。当前没有接相机时，先用测试文件占位。"
    # cap = cv2.VideoCapture(0)
    # ret, frame = cap.read()
    # if ret:
    #     image_path = image_dir / f"img_{idx:03d}.png"
    #     cv2.imwrite(str(image_path), frame)
    # cap.release()

    image_path = image_dir / f"img_{idx:03d}.png"
    frame = cv2.imread(str(PROJECT_ROOT / "leather" / "good" / "000.png"))
    if frame is not None:
        cv2.imwrite(str(image_path), frame)
    else:
        image_path.write_text("test image placeholder\n", encoding="utf-8")
    print("保存图片:", image_path)


def ImageZip(version_root, image_dir):
    "压缩本次采集的图片"
    zip_path = version_root / f"{version_root.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in image_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(version_root)
                zf.write(file_path, arcname)
    print("打包完成:", zip_path)
    return zip_path


def UploadServer(zip_path):
    "把图片包上传到 PC"
    upload_url = f"{PC_URL}/upload"
    print("开始上传:", upload_url)
    with open(zip_path, "rb") as f:
        files = {"file": (zip_path.name, f, "application/zip")}
        response = requests.post(upload_url, files=files, timeout=60)
    response.raise_for_status()
    print("上传完成:", response.text)


def CheckModelReady(folder_name):
    "询问 PC 端模型是否训练完成。PC 未实现 ready 接口时，直接尝试下载。"
    ready_url = f"{PC_URL}/ready/{folder_name}"
    try:
        response = requests.get(ready_url, timeout=10)
        if response.status_code == 404:
            return True
        response.raise_for_status()
        data = response.json()
        return bool(data.get("ready", False))
    except Exception as exc:
        print("检查训练状态失败，稍后重试:", exc)
        return False


def DownloadModel(folder_name, model_dir):
    "下载模型到新版本目录，先写 tmp，成功后再 rename"

    download_url = f"{PC_URL}/download/{folder_name}"
    tmp_path = model_dir / f"{MODEL_NAME}.tmp"
    model_path = model_dir / MODEL_NAME

    print("开始下载模型:", download_url)
    response = requests.get(download_url)
    # 在请求失败的时候抛出 HTTPError 异常
    response.raise_for_status()

    with open(tmp_path, "wb") as f:
        f.write(response.content)

    if tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("下载到的模型文件为空")

    CheckModel(tmp_path)
    tmp_path.replace(model_path)
    print("模型下载完成:", model_path)
    return model_path


def CheckModel(model_path):
    "校验 ONNX 是否能被加载"
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    print("模型校验通过:", model_path)


def MarkReady(model_dir, folder_name):
    "写入模型版本信息"
    meta_path = model_dir / "meta.txt"
    meta_path.write_text(
        f"folder_name={folder_name}\nupdated_at={datetime.now().isoformat()}\n",
        encoding="utf-8",
    )


def SwitchCurrentModel(model_dir):
    "把 current_model 切到新模型目录"
    tmp_link = PROJECT_ROOT / "current_model.tmp"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()

    tmp_link.symlink_to(model_dir, target_is_directory=True)
    tmp_link.replace(CURRENT_MODEL_LINK)
    print("当前模型已切换到:", model_dir)
    print("当前模型路径:", CURRENT_MODEL_LINK / MODEL_NAME)


def UpdateModel():
    folder_name, version_root, image_dir, model_dir = NewFolder()

    for idx in range(TARGET_IMAGE_COUNT):
        # 捕捉图像
        CaptureImages(image_dir, idx)

    zip_path = ImageZip(version_root, image_dir)
    # 上传到服务器
    UploadServer(zip_path)
    DownloadModel(folder_name, model_dir)
    MarkReady(model_dir, folder_name) # 写入信息
    SwitchCurrentModel(model_dir)


def main():
    if not CheckNewFabric():
        print("不是新布，不更新模型")
        return

    # 更新模型
    UpdateModel()


if __name__ == "__main__":
    main()
