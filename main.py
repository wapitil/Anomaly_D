from YOLOv5_Detect import start_detect
from deviceCtrl import start_ctrl
from myLogger import logger
from network import test

if __name__ == '__main__':
    logger.info('start app')
    # test()
    start_detect()
    start_ctrl()
