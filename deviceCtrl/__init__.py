import Hobot.GPIO as GPIO
import threading
import sys
import signal
import os
import time
from enum import Enum
# 导入python串口库
import serial
import serial.tools.list_ports
try:
    from myLogger import logger
except ImportError:
    import logging
    logger = logging.getLogger('logger')
    logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s',
	)

class Direction(Enum):
    """移动方向枚举"""
    OPEN = "open"   # 向左移动
    CLOSE = "close" # 向右移动

class DeviceCtrl:
    autoctrl_btn = False
    now_type = None
    position = {
        Direction.OPEN.value: "机器到达最左边",
        Direction.CLOSE.value: "机器到达最右边"
    }
    def __init__(self):
        # GPIO.cleanup()
        GPIO.setmode(GPIO.BOARD)
        self.wake_index = 32
        # 设置32引脚输出 行走MCU唤醒
        GPIO.setup(self.wake_index, GPIO.OUT, initial=GPIO.HIGH)


        self.flashlight_index = 37
        GPIO.setup(self.flashlight_index, GPIO.OUT, initial=GPIO.HIGH)
        # 设置37引脚输出 闪光灯控制引脚 默认开启

        self.shift_index = 3
        GPIO.setup(self.shift_index, GPIO.IN)
        # 设置3脚输入 左右限位器信号

        self.ser = serial.Serial('/dev/ttyS2', 115200, timeout=1)
        
        self.opendata  = '55AA05010101EAA9'
        self.closedata = '55AA05010102EAA3'
        self.stopdata  = '55AA050101036AA6'

        self.now_type = 'stop'
        self.autoctrl_btn = False

    def open(self):
        self.now_type = 'open'
        self._wake()
        self.ser.write(bytes.fromhex(self.opendata))

    def close(self):
        self.now_type = 'close'
        self._wake()
        self.ser.write(bytes.fromhex(self.closedata))

    def stop(self):
        # self.now_type = 'stop'
        self._wake()
        self.ser.write(bytes.fromhex(self.stopdata))


    def autoctrl_(self):
        """
        自动往返运动控制主逻辑
        工作流程：
        1. 启动时先移动到左侧(open)
        2. 每秒循环检测位置信号
        3. 当检测到到达信号(shift=0)且满足最小运行时间，切换方向
        4. 切换时执行：停止 → 延时 → 反向启动
        """
        # ========== 配置参数 ==========
        MIN_RUN_TIME = 5        # 切换方向最小间隔时间（秒）
        STOP_CYCLES = 5         # 停止信号持续周期数（用于确保停稳）
        CYCLE_INTERVAL = 0.02    # 主循环周期（秒）
        LOG_INTERVAL = 1.0      # 日志打印间隔（秒）
        
        # ========== 初始化 ==========
        self.open()  # 启动时先向左移动
        self.now_type = Direction.OPEN.value
        
        last_direction_change = time.time() - MIN_RUN_TIME  # 初始允许立即切换
        last_log_time = time.time()
        # cycle_counter = 0
        stop_countdown = 0
        
        logger.info("自动往返运动已启动")
        
        # ========== 主循环 ==========
        while self.autoctrl_btn:
            # cycle_counter += 1
            current_time = time.time()
            
            # --- 每秒打印状态日志 ---
            if current_time - last_log_time >= LOG_INTERVAL:
                # logger.info(f"运行状态: 方向={self.now_type}, 周期数={cycle_counter}")
                logger.info(f"运行状态: 方向={self.now_type}")
                # cycle_counter = 0
                last_log_time = current_time
            
            # --- 检测位置信号并决定方向 ---
            position_signal = self._get_shift()  # 0=到达, 1=运行中
            
            if position_signal == 0:
                time_since_last_change = current_time - last_direction_change
                
                # 判断是否满足切换条件
                if time_since_last_change >= MIN_RUN_TIME:
                    # 切换方向
                    new_direction = (
                        Direction.CLOSE.value if self.now_type == Direction.OPEN.value 
                        else Direction.OPEN.value
                    )
                    logger.info(f"位置到达: {self.position[self.now_type]} → 切换方向至 {new_direction}")
                    
                    self.now_type = new_direction
                    last_direction_change = current_time
                    stop_countdown = STOP_CYCLES  # 启动停止序列
            
            # --- 执行停止/启动命令 ---
            if stop_countdown == STOP_CYCLES:
                # 立即停止
                self.stop()
                logger.debug("执行停止命令")
                stop_countdown -= 1
            elif stop_countdown > 0:
                # 停止中，等待设备稳定
                stop_countdown -= 1
            else:
                # 正常运行
                if self.now_type == Direction.OPEN.value:
                    self.open()
                elif self.now_type == Direction.CLOSE.value:
                    self.close()
                # else:
                #     logger.error(f'{self.now_type=}')
            
            time.sleep(CYCLE_INTERVAL)
        
        logger.info("自动往返运动已停止")


    def autoctrl(self):  # open 代表向左走， close代表向右走
        self.position = {'open':'机器到达最左边', 'close':'机器到达最右边'}
        self.open()
        last_change_time = time.time() - 5
        # 新增：方向切换后的保护时间（至少运行X秒后才允许再次切换）
        MIN_RUN_TIME = 5  # 最少运行5秒才能再次切换
        # 标记是否已处理过当前磁铁信号
#        is_processed = False  
        second = time.time()
        # shift_second = time.time()
        index = 0
        shift = 1
        while self.autoctrl_btn:
            index += 1
            if time.time() - second > 1:
                # logger.info(f'one second get shift = {index}')
                index = 0
                second = time.time()

            shift = self._get_shift()
            if time.time() - last_change_time > 40:
                shift = 0
                # last_change_time = time.time()
            # else:
            #     shift = 1
            
            if shift == 0:
                # logger.info(f'{shift=}')
                # logger.info(f'{shift=}')
                # 增加保护时间检查：确保切换后至少运行了MIN_RUN_TIME秒
                time_since_last_change = time.time() - last_change_time
#                if (not is_processed and 
#                    time_since_last_change >= MIN_RUN_TIME):
                if (time_since_last_change >= MIN_RUN_TIME):
                    logger.info(f'position: {self.position[self.now_type]}')
                    self.now_type = 'open' if self.now_type=='close' else 'close'
                    last_change_time = time.time()
                    logger.info(f'{self.now_type=}')
#                    is_processed = True  # 标记为已处理
#            else:
#                is_processed = False  # 离开磁铁区域，重置标记

            # 执行当前状态操作

            if self.now_type == 'open':
                self.open()
            else:
                self.close()
                    
            time.sleep(0.01)

    def _get_shift(self):
        return GPIO.input(self.shift_index)

    def _wake(self):

        GPIO.output(self.wake_index, GPIO.LOW)

        time.sleep(0.02)
        GPIO.output(self.wake_index, GPIO.HIGH)


    def exit(self):
        GPIO.output(self.flashlight_index, GPIO.LOW)
        GPIO.cleanup(self.flashlight_index)
    
    def getSer(self):
        msg = ''
        while True:
            tmp = self.ser.read().hex()
            if tmp:
                msg += tmp + ' '
            elif msg:
                logger.info(f'{msg=}')
#                logger.info(f'{msg=}')
                msg = ''

    def sendMsg(self):
        # return
        pass
        # while True:
        #     if self.now_type == 'open':
        #         self.open()
        #     elif self.now_type == 'close':
        #         self.close()
        #     else:
        #         self.stop()
        #     time.sleep(0.02)

dc = DeviceCtrl()
def __start_ctrl():
    dc.stop()
    dc.open()
    time.sleep(40)
    dc.close()
    time.sleep(10)
    dc.stop()
    dc.autoctrl_btn = True
    th = threading.Thread(target=dc.autoctrl, daemon=True)
    th.start()

def start_ctrl():
    thread = threading.Thread(target=__start_ctrl)
    thread.start()

if __name__ == '__main__':
    
    th = threading.Thread(target=dc.sendMsg, daemon=True)
    th.start()
    th1 = threading.Thread(target=dc.getSer, daemon=True)
    th1.start()


    while True:
        try:
            res = int(input('1: 开\t2: 关\t3: 停\t4: 自动控制\t5. 停止自动控制\t6. 唤醒\t0: 退出'))
        except:
            continue
        dc.autoctrl_btn = False
        if res == 1:
            dc.open()
        elif res == 2:
            dc.close()
        elif res == 3:
            dc.stop()
        elif res == 4:
            dc.autoctrl_btn = True
            th = threading.Thread(target=dc.autoctrl, daemon=True)
            th.start()
        elif res == 5:
            dc.autoctrl_btn = False
            dc.stop()
        elif res == 6:
            dc._wake()
        elif res == 0:
            
            dc.exit()
            break
        print('操作完成')
        



