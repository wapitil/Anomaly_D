import numpy as np
import cv2
from utils import parse
from datetime import datetime
from myLogger import logger
import base64
import httpx

# numpy 转 base64
def numpy_to_base64(image_np): 
    data = cv2.imencode('.jpg', image_np)[1]
    image_bytes = data.tobytes()
    image_base4 = base64.b64encode(image_bytes).decode('utf8')
    return image_base4

async def sendDefect(image_np, defect_name):
    server_url = parse['server_url']
    defect_data = {
        "deviceID": parse['device_id'],
        "imageBase64": numpy_to_base64(image_np),
        "defectName": defect_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "severity": '高'
    }
    headers = {
        "Content-Type": "application/json",
        "X-API-Token": parse['api_token']
    }
    # logger.info(f'{defect_data=}')
    # logger.info(f'{headers=}')
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{server_url}/api/defect",
                json=defect_data,
                headers=headers,
                timeout=10.0
            )

            if response.status_code == 200:
                result = response.json()
                queued = result.get("data", {}).get("queued", False)
                image_url = result.get("data", {}).get("imageUrl", "")
                if queued:
                    logger.info(f"已加入离线队列 (设备离线)")
                else:
                    logger.info(f"实时推送成功 (设备在线)")
                logger.info(f"图像: {image_url}")
            else:
                logger.info(f"上报失败: {response.status_code}")
                logger.info(f"响应: {response.text}")

    except Exception as e:
        logger.info(f"上报异常: {e}")


def __sendDefect(image_np, defect_name):
    thread = threading.Thread(target=__sendDefect, args=(image_np, defect_name, ))
    thread.start()

def test():
    import asyncio
    from utils import class_id2name
    out_img = cv2.imread('/home/sunrise/Pictures/v.png')
    ids = [62]
    try:
        # await sendDefect(out_img, class_id2name(ids))
        pass
    except Exception as e:
        logger.error(e)
        pass
    asyncio.run(sendDefect(out_img, class_id2name(ids)))