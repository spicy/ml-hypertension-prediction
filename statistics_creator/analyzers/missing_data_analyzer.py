import pandas as pd
from .base_analyzer import BaseAnalyzer
from logger import logger, log_execution_time

class MissingDataAnalyzer(BaseAnalyzer):
    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> pd.Series:
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        missing_percentage_sorted = missing_percentage.sort_values(ascending=False)
        logger.info("Missing data analysis completed.")
        return missing_percentage_sorted