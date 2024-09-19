import logging
import time
import os
from functools import wraps
from typing import Callable, Any

def setup_logger() -> logging.Logger:
    """
    Sets up and configures the logger.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger('statistics_creator')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logs_folder = 'logs'
    os.makedirs(logs_folder, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(logs_folder, 'statistics_creator.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def log_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator that logs the execution time of a function.

    Args:
        func (Callable[..., Any]): The function to be decorated.

    Returns:
        Callable[..., Any]: The wrapped function.
    """
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