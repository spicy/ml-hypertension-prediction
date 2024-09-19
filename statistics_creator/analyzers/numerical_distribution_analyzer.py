import pandas as pd
import numpy as np
from scipy import stats
from .base_analyzer import BaseAnalyzer
from logger import logger, log_execution_time

class NumericalDistributionAnalyzer(BaseAnalyzer):
    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        results = {}

        for column in numeric_columns:
            column_data = df[column].dropna()
            if len(column_data) > 0:
                results[column] = {
                    'mean': column_data.mean(),
                    'median': column_data.median(),
                    'std': column_data.std(),
                    'skewness': stats.skew(column_data),
                    'kurtosis': stats.kurtosis(column_data),
                    'min': column_data.min(),
                    'max': column_data.max(),
                    'q1': column_data.quantile(0.25),
                    'q3': column_data.quantile(0.75),
                }

        logger.info("Numerical distribution analysis completed.")
        return results