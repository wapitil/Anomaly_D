
import logging.config
import os
import colorlog

# 获取当前文件的绝对路径所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 构造日志目录路径
base_log_dir = os.path.join(BASE_DIR, "log")
info_log_dir = os.path.join(base_log_dir, "info")
error_log_dir = os.path.join(base_log_dir, "error")

# 创建目录
for path in [info_log_dir, error_log_dir]:
    os.makedirs(path, exist_ok=True)


# 日志配置字典
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,  # 保留已经配置好的 logger 设置，同时加载你自己新增的日志配置
    'formatters': {
        'standard_formatter': {  # 标准输出
            'format': '[%(asctime)s.%(msecs)03d][%(pathname)s line:%(lineno)d][%(funcName)s]'
                      '[%(threadName)s: %(thread)d][%(levelname)s] %(message)s',
            'datefmt': "%Y-%m-%d %H:%M:%S"
        },
        'color_formatter': {  # 颜色格式化标准输出
            '()': colorlog.ColoredFormatter,
            'format': '%(log_color)s[%(asctime)s.%(msecs)03d][%(pathname)s line:%(lineno)d]'
                      '[%(funcName)s][%(threadName)s: %(thread)d][%(levelname)s] %(message)s',
            'datefmt': "%Y-%m-%d %H:%M:%S",
            'log_colors': {
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        },
    },
    'filters': {},  # 日志过滤器
    'handlers': {
        'console': {  # 输出到日志控制台
            'level': 'INFO',  # 日志等级
            'class': 'logging.StreamHandler',  # 输出到控制台
            'formatter': 'color_formatter',  # 颜色格式化标准输出
        },
        'file_info': {  # 输出到文件
            'level': 'INFO',  # 日志等级
            'formatter': 'standard_formatter',  # 标准输出
            'class': "logging.handlers.TimedRotatingFileHandler",  # 按日期自动轮转日志，保留历史记录
            'filename': os.path.join(info_log_dir, "info.log"),  # 输出日志的文件
            'when': 'D',  # 按天分割
            'interval': 1,  # 间隔1天
            'backupCount': 5,  # 保留天数
            'encoding': 'utf-8',  # utf-8编码
        },
        'file_error': {
            'level': 'ERROR',
            'formatter': 'standard_formatter',
            'class': "logging.handlers.TimedRotatingFileHandler",
            'filename': os.path.join(error_log_dir, "error.log"),
            'when': 'D',
            'interval': 1,
            'backupCount': 5,
            'encoding': 'utf-8'
        }
    },
    'loggers': {  # 根据loggers.getLogger(__name__)获取logger配置
        'logger': {
            'handlers': ['console', 'file_info', 'file_error'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('logger')