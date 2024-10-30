import logging
import os
import time
from functools import wraps
from typing import Any, Callable

import colorlog
from config import logger_config


def setup_logger():
    """Set up colored logging with both file and console handlers."""
    logger = colorlog.getLogger(logger_config.LOGGER_NAME)
    logger.setLevel(logging.INFO)

    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            logger_config.LOG_FORMAT, log_colors=logger_config.LOG_COLORS
        )
    )
    logger.addHandler(console_handler)

    # File handler
    os.makedirs(logger_config.LOGS_FOLDER, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(logger_config.LOGS_FOLDER, logger_config.LOG_FILE_NAME)
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    return logger


def log_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that logs the execution time of a function."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result

    return wrapper


logger = setup_logger()
