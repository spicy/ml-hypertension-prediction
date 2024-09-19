import os
import pandas as pd
from logger import logger, log_execution_time

class DataLoader:
    @staticmethod
    def create_statistics_folder(folder_path: str = None) -> str:
        if folder_path is None:
            folder_path = os.path.join(os.getcwd(), 'statistics')

        logger.info("Creating statistics folder...")
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"Statistics folder created at: {folder_path}")
        return folder_path

    @staticmethod
    @log_execution_time
    def load_data(file_path: str) -> pd.DataFrame:
        logger.info(f"Loading data from {file_path}...")

        if not os.path.exists(file_path):
            logger.error(f"The file '{file_path}' does not exist.")
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        try:
            with open(file_path, 'r') as file:
                df = pd.read_csv(file)
            if df.empty:
                logger.error(f"The file '{file_path}' is empty.")
                raise ValueError(f"The file '{file_path}' is empty.")
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except pd.errors.ParserError as e:
            logger.error(f"Unable to parse '{file_path}': {e}")
            raise ValueError(f"Unable to parse '{file_path}': {e}")