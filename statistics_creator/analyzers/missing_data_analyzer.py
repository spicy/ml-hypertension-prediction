import pandas as pd
from .base_analyzer import BaseAnalyzer
from logger import logger, log_execution_time

class MissingDataAnalyzer(BaseAnalyzer):
    """
    A class for analyzing missing data in a DataFrame.
    """

    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates the percentage of missing values for each column
        in the DataFrame and returns a sorted Series of these percentages.

        Args:
            df (pd.DataFrame): The input DataFrame to analyze.

        Returns:
            pd.Series: A Series containing the percentage of missing values
                       for each column, sorted in descending order.
        """
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        missing_percentage_sorted = missing_percentage.sort_values(ascending=False)
        logger.info("Missing data analysis completed.")
        return missing_percentage_sorted