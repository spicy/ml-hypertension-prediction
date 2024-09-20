import pandas as pd
import numpy as np
from .base_analyzer import BaseAnalyzer
from logger import logger, log_execution_time

class OutlierAnalyzer(BaseAnalyzer):
    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> dict:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outliers = {}

        for column in numeric_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers[column] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers': df[(df[column] < lower_bound) | (df[column] > upper_bound)][column].tolist()
            }

        logger.info("Outlier detection completed.")
        return outliers