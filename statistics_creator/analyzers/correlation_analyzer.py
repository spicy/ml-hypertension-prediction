import pandas as pd
import numpy as np
from .base_analyzer import BaseAnalyzer
from logger import logger, log_execution_time

class CorrelationAnalyzer(BaseAnalyzer):
    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        logger.info("Correlation matrix calculation completed.")
        return correlation_matrix