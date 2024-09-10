import os
import pandas as pd
from logger import logger, log_execution_time

def create_statistics_folder() -> str:
    logger.info("Creating statistics folder...")
    statistics_folder = os.path.join(os.getcwd(), 'statistics')
    os.makedirs(statistics_folder, exist_ok=True)
    logger.info(f"Statistics folder created at: {statistics_folder}")
    return statistics_folder

@log_execution_time
def load_data(file_path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        logger.error(f"The file '{file_path}' does not exist.")
        raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except pd.errors.EmptyDataError:
        logger.error(f"The file '{file_path}' is empty.")
        raise ValueError(f"Error: The file '{file_path}' is empty.")
    except pd.errors.ParserError:
        logger.error(f"Unable to parse '{file_path}'. Please check if it's a valid CSV file.")
        raise ValueError(f"Error: Unable to parse '{file_path}'. Please check if it's a valid CSV file.")