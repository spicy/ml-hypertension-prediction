import numpy as np
import pandas as pd
from config import outlier_config as config
from logger import log_execution_time, logger

from .base_analyzer import BaseAnalyzer


class OutlierAnalyzer(BaseAnalyzer):
    """
    A class for analyzing and detecting outliers in a DataFrame.
    """

    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Analyze the input DataFrame and detect outliers for numeric columns.

        It uses the Interquartile Range method to identify outliers.
        An outlier is defined as any value below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR,
        where Q1 is the first quartile, Q3 is the third quartile, and IQR is the interquartile range.
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outliers = {}

        for column in numeric_columns:
            Q1 = df[column].quantile(config.LOWER_QUANTILE)
            Q3 = df[column].quantile(config.UPPER_QUANTILE)
            IQR = Q3 - Q1
            lower_bound = Q1 - config.IQR_MULTIPLIER * IQR
            upper_bound = Q3 + config.IQR_MULTIPLIER * IQR

            outliers[column] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outliers": df[(df[column] < lower_bound) | (df[column] > upper_bound)][
                    column
                ].tolist(),
            }

        logger.info("Outlier detection completed.")
        return outliers
