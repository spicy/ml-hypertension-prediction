import os

import pandas as pd
from config import data_config
from logger import log_execution_time, logger


class DataLoader:
    @staticmethod
    def create_statistics_folder(folder_path: str = None) -> str:
        """
        Create a folder to store statistics.
        """
        if folder_path is None:
            folder_path = os.path.join(
                os.getcwd(), data_config.DEFAULT_STATISTICS_FOLDER
            )

        logger.info(f"Creating statistics folder at {folder_path}...")
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"Statistics folder created at: {folder_path}")
        return folder_path

    @staticmethod
    @log_execution_time
    def load_data(file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file into a pandas DataFrame.
        """
        logger.info(f"Loading data from {file_path}...")

        if not os.path.exists(file_path):
            error_msg = f"The file '{file_path}' does not exist."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            df = pd.read_csv(file_path)
            if df.empty:
                error_msg = f"The file '{file_path}' is empty."
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except pd.errors.ParserError as e:
            error_msg = f"Unable to parse '{file_path}': {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
