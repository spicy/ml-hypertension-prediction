import pandas as pd
from scipy import stats
from .base_analyzer import BaseAnalyzer
from config import comprehensive_numerical_config as config
from logger import logger, log_execution_time

class ComprehensiveNumericalAnalyzer(BaseAnalyzer):
    """
    A class for performing comprehensive numerical analysis on a DataFrame,
    including basic descriptive statistics, skewness, and kurtosis.
    """

    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Analyze the input DataFrame to calculate comprehensive numerical statistics.

        It performs the following steps:
        1. Selects numeric columns from the input DataFrame.
        2. For each numeric column, calculates basic descriptive statistics,
           skewness, and kurtosis.
        3. Combines all statistics into a single dictionary.

        Args:
            df (pd.DataFrame): The input DataFrame to analyze.

        Returns:
            dict: A dictionary containing comprehensive numerical statistics
                  for each numeric column in the input DataFrame.

        Note:
            Only non-null values are considered in the analysis for each column.
        """
        logger.info("Starting comprehensive numerical analysis...")

        numeric_columns = df.select_dtypes(include=config.NUMERIC_DTYPES).columns
        results = {}

        for column in numeric_columns:
            column_data = df[column].dropna()
            if len(column_data) > 0:
                stats_dict = column_data.describe().to_dict()
                stats_dict.update({
                    config.SKEWNESS_KEY: stats.skew(column_data),
                    config.KURTOSIS_KEY: stats.kurtosis(column_data),
                })
                results[column] = stats_dict

        logger.info("Comprehensive numerical analysis completed.")
        return results