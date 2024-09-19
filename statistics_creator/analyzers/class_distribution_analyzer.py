import pandas as pd
from .base_analyzer import BaseAnalyzer
from logger import logger, log_execution_time

class ClassDistributionAnalyzer(BaseAnalyzer):
    def __init__(self, target_column: str):
        self.target_column = target_column

    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> pd.Series:
        if self.target_column not in df.columns:
            logger.error(f"Target column '{self.target_column}' not found in the dataset.")
            raise ValueError(f"Target column '{self.target_column}' not found in the dataset.")

        target_values = df[self.target_column].dropna()
        value_counts = target_values.value_counts()
        total = len(target_values)

        logger.info(f"Class distribution calculated for column '{self.target_column}'")
        return (value_counts / total) * 100