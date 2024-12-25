import glob
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
    def load_data(file_path: str = None) -> pd.DataFrame:
        """
        Load data from CSV file(s) into a pandas DataFrame.
        If file_path is provided, loads a single file.
        If no file_path is provided, loads and combines all autofilled data files.
        """
        if file_path:
            return DataLoader._load_single_file(file_path)
        else:
            return DataLoader._load_autofilled_data()

    @staticmethod
    def _load_single_file(file_path: str) -> pd.DataFrame:
        """
        Load a single CSV file into a DataFrame.
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

    @staticmethod
    def _load_autofilled_data() -> pd.DataFrame:
        """
        Load and combine all autofilled data files.
        """
        pattern = os.path.join(
            data_config.AUTOFILLED_DIR, data_config.FILTERED_DATA_PATTERN
        )
        files = glob.glob(pattern)

        if not files:
            error_msg = f"No files found matching pattern: {pattern}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        data_frames = []
        for file in sorted(files):
            df = pd.read_csv(file, index_col="SEQN")
            data_frames.append(df)

        # Combine all the loaded dataframes into one
        data = pd.concat(data_frames, axis=0)

        # Change the HYPERTENSION column to int
        data["HYPERTENSION"] = data["HYPERTENSION"].astype(int)

        logger.info(f"Combined data loaded successfully. Shape: {data.shape}")
        return data
