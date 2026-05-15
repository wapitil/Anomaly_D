#!/usr/bin/env python3
"""
硬件访问测试脚本
用于验证 Docker 容器内是否能正常访问硬件和库
"""

import sys
import os

def test_hobot_dnn():
    """测试 hobot_dnn 库"""
    try:
        from hobot_dnn import pyeasy_dnn as dnn
        print("✅ hobot_dnn 导入成功")
        return True
    except ImportError as e:
        print(f"❌ hobot_dnn 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ hobot_dnn 测试失败: {e}")
        return False

def test_hobot_gpio():
    """测试 Hobot.GPIO 库"""
    try:
        import Hobot.GPIO as GPIO
        print("✅ Hobot.GPIO 导入成功")
        return True
    except ImportError as e:
        print(f"❌ Hobot.GPIO 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ Hobot.GPIO 测试失败: {e}")
        return False

def test_opencv():
    """测试 OpenCV"""
    try:
        import cv2
        print(f"✅ OpenCV 导入成功 (版本: {cv2.__version__})")
        return True
    except Exception as e:
        print(f"❌ OpenCV 测试失败: {e}")
        return False

def test_camera():
    """测试摄像头访问"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                print(f"✅ 摄像头访问成功 (分辨率: {frame.shape[1]}x{frame.shape[0]})")
                return True
            else:
                print("⚠️  摄像头已打开但无法读取画面")
                return False
        else:
            print("❌ 摄像头无法打开")
            return False
    except Exception as e:
        print(f"❌ 摄像头测试失败: {e}")
        return False

def test_serial():
    """测试串口访问"""
    try:
        import serial
        # 尝试打开串口（不进行实际通信）
        try:
            ser = serial.Serial('/dev/ttyS2', 115200, timeout=1)
            ser.close()
            print("✅ 串口 /dev/ttyS2 访问成功")
            return True
        except serial.SerialException as e:
            print(f"⚠️  串口 /dev/ttyS2 访问失败: {e}（可能设备未连接）")
            return False
    except ImportError:
        print("❌ pyserial 库未安装")
        return False
    except Exception as e:
        print(f"❌ 串口测试失败: {e}")
        return False

def test_gpio_device():
    """测试 GPIO 设备访问"""
    try:
        # 检查 GPIO 设备文件是否存在
        gpio_devices = ['/dev/gpiochip0', '/dev/gpiochip1', '/sys/class/gpio']
        found = []
        for device in gpio_devices:
            if os.path.exists(device):
                found.append(device)

        if found:
            print(f"✅ GPIO 设备可访问: {', '.join(found)}")
            return True
        else:
            print("⚠️  未找到 GPIO 设备文件")
            return False
    except Exception as e:
        print(f"❌ GPIO 设备测试失败: {e}")
        return False

def test_model_file():
    """测试模型文件是否存在"""
    try:
        import yaml
        with open('/app/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        model_path = config.get('model_path', '/app/models/model.bin')

        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✅ 模型文件存在: {model_path} ({size_mb:.2f} MB)")
            return True
        else:
            print(f"⚠️  模型文件不存在: {model_path}")
            return False
    except Exception as e:
        print(f"⚠️  模型文件测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("=" * 50)
    print("硬件访问测试 - Docker 容器内")
    print("=" * 50)
    print()

    results = []

    # 核心库测试
    print("【1/7】测试 hobot_dnn...")
    results.append(("hobot_dnn", test_hobot_dnn()))
    print()

    print("【2/7】测试 Hobot.GPIO...")
    results.append(("Hobot.GPIO", test_hobot_gpio()))
    print()

    print("【3/7】测试 OpenCV...")
    results.append(("OpenCV", test_opencv()))
    print()

    # 硬件设备测试
    print("【4/7】测试摄像头...")
    results.append(("摄像头", test_camera()))
    print()

    print("【5/7】测试串口...")
    results.append(("串口", test_serial()))
    print()

    print("【6/7】测试 GPIO 设备...")
    results.append(("GPIO 设备", test_gpio_device()))
    print()

    print("【7/7】测试模型文件...")
    results.append(("模型文件", test_model_file()))
    print()

    # 总结
    print("=" * 50)
    print("测试结果总结")
    print("=" * 50)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:15s} : {status}")

    print()
    print(f"总计: {passed}/{total} 测试通过")
    print("=" * 50)

    # 返回退出码（0=全部通过，1=有失败）
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
