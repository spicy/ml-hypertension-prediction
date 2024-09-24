import pandas as pd
import numpy as np
from .base_analyzer import BaseAnalyzer
from logger import logger, log_execution_time

class OutlierAnalyzer(BaseAnalyzer):
    """
    A class for analyzing and detecting outliers in a DataFrame.
    """

    LOWER_QUANTILE = 0.25
    UPPER_QUANTILE = 0.75
    IQR_MULTIPLIER = 1.5

    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Analyze the input DataFrame and detect outliers for numeric columns.

        It uses the Interquartile Range method to identify outliers.
        An outlier is defined as any value below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR,
        where Q1 is the first quartile, Q3 is the third quartile, and IQR is the interquartile range.

        Args:
            df (pd.DataFrame): The input DataFrame to analyze.

        Returns:
            dict: A dictionary containing outlier information for each numeric column.
                  The keys are column names, and the values are dictionaries with:
                  - 'lower_bound': The lower threshold for outliers.
                  - 'upper_bound': The upper threshold for outliers.
                  - 'outliers': A list of outlier values found in the column.

        Note:
            It only analyzes numeric columns in the input DataFrame.
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outliers = {}

        for column in numeric_columns:
            Q1 = df[column].quantile(self.LOWER_QUANTILE)
            Q3 = df[column].quantile(self.UPPER_QUANTILE)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.IQR_MULTIPLIER * IQR
            upper_bound = Q3 + self.IQR_MULTIPLIER * IQR

            outliers[column] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers': df[(df[column] < lower_bound) | (df[column] > upper_bound)][column].tolist()
            }

        logger.info("Outlier detection completed.")
        return outliers