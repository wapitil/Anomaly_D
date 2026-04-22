import logging
from pathlib import Path

import colorlog


def setup_logger():
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
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

    logger = logging.getLogger()  # 注意：这里不要写 __name__
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        logger.addHandler(handler)


def check_sim():
    """检查转换的软连接是否正确"""
    good_dir = Path("/home/wapiti/Projects/Anomaly_D/Server/images/20260422_141516")

    for p in list(good_dir.iterdir())[:20]:
        print("name:", p.name)
        print("is_symlink:", p.is_symlink())
        print("exists:", p.exists())  # 目标文件存在才会 True
        print("resolve:", p.resolve(strict=False))
        print("-" * 40)
