import pandas as pd
import numpy as np
from .base_analyzer import BaseAnalyzer
from logger import logger, log_execution_time

class CorrelationAnalyzer(BaseAnalyzer):
    """
    A class for analyzing correlations between numeric variables in a DataFrame.
    """

    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze the input DataFrame to calculate correlations between numeric variables.

        It performs the following steps:
        1. Selects only numeric columns from the input DataFrame.
        2. Calculates the correlation matrix for these numeric columns.
        3. Logs the completion of the correlation matrix calculation.

        Args:
            df (pd.DataFrame): The input DataFrame to analyze.

        Returns:
            pd.DataFrame: A DataFrame containing the correlation matrix of numeric variables.

        Note:
            Only numerical columns are considered in this analysis.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        logger.info("Correlation matrix calculation completed.")
        return correlation_matrix