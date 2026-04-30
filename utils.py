import logging

import colorlog


def setup_logger(level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        return

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )
    root.addHandler(handler)


setup_logger()
