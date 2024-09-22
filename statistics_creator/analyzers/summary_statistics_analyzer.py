import pandas as pd
from .base_analyzer import BaseAnalyzer
from logger import logger, log_execution_time

class SummaryStatisticsAnalyzer(BaseAnalyzer):
    """
    A class for analyzing and generating summary statistics for a given DataFrame.
    """

    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze the input DataFrame and generate summary statistics.

        It computes descriptive statistics including count, mean,
        standard deviation, minimum, 25th percentile, median, 75th percentile,
        and maximum for all numerical columns in the input DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame to analyze.

        Returns:
            pd.DataFrame: A DataFrame containing the summary statistics for each numerical column.

        Note:
            The returned DataFrame includes statistics for numerical columns only.
            Categorical columns are excluded from the analysis.
        """
        summary_statistics = df.describe()
        logger.info("Summary statistics generation completed.")
        return summary_statistics