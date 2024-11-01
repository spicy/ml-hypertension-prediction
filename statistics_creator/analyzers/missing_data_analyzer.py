import pandas as pd
from config import missing_data_config as config
from logger import log_execution_time, logger

from .base_analyzer import BaseAnalyzer


class MissingDataAnalyzer(BaseAnalyzer):
    """
    A class for analyzing missing data in a DataFrame.
    """

    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates the percentage of missing values for each column
        in the DataFrame and returns a sorted Series of these percentages.
        """
        missing_percentage = (
            df.isnull().sum() / len(df)
        ) * config.PERCENTAGE_MULTIPLIER
        missing_percentage_sorted = missing_percentage.sort_values(ascending=False)
        logger.info(config.MISSING_DATA_LOG_MESSAGE)
        return missing_percentage_sorted
