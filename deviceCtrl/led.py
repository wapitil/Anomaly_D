import time
import os
import Hobot.GPIO as GPIO

STATE_FILE = "/tmp/led_state.txt"

try:
    from myLogger import logger
except ImportError:
    import logging
    logger = logging.getLogger("logger")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LightCtrl:
    """GPIO 37 LED controller"""

    def __init__(self, pin=37):
        self.pin = pin
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
        logger.info("LED initialized on BOARD pin %s", self.pin)

        # 读取上次状态
        self.state = False
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                content = f.read().strip()
                if content == "1":
                    self.state = True
                    self.on()
                else:
                    self.state = False
                    self.off()

    def on(self):
        GPIO.output(self.pin, GPIO.HIGH)
        self.state = True
        self._save_state()
        logger.info("LED ON")

    def off(self):
        GPIO.output(self.pin, GPIO.LOW)
        self.state = False
        self._save_state()
        logger.info("LED OFF")

    def toggle(self):
        if self.state:
            self.off()
        else:
            self.on()

    def _save_state(self):
        with open(STATE_FILE, "w") as f:
            f.write("1" if self.state else "0")

    def cleanup(self):
        self.off()
        GPIO.cleanup(self.pin)
        logger.info("LED GPIO cleaned up")


light = LightCtrl()


def led_on():
    light.on()


def led_off():
    light.off()


def led_toggle():
    light.toggle()


def exit():
    light.cleanup()


if __name__ == "__main__":
    try:
        # 每次运行就切换状态
        led_toggle()
    finally:
        GPIO.cleanup()
