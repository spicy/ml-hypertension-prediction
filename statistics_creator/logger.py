import logging
import time
import os
from functools import wraps

def setup_logger():
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

logger = setup_logger()

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper