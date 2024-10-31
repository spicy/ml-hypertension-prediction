import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict

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


def setup_detailed_logger(
    name: str,
    log_level: str = "DEBUG",
    log_format: str = None,
    log_colors: Dict[str, str] = None,
) -> logging.Logger:
    """Setup a detailed logger with color support and structured formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    if not logger.handlers:
        # Console handler with colors
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s%(reset)s",
                log_colors=log_colors
                or {
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
        )
        logger.addHandler(console_handler)

        # File handler for detailed logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}_detailed.log")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s\n"
                "Additional Context:\n"
                "Function: %(funcName)s\n"
                "Path: %(pathname)s\n"
                "Process: %(process)d\n"
                "---\n"
            )
        )
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()
