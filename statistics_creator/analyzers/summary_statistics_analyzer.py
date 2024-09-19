import pandas as pd
from .base_analyzer import BaseAnalyzer
from logger import logger, log_execution_time

class SummaryStatisticsAnalyzer(BaseAnalyzer):
    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        summary_statistics = df.describe()
        logger.info("Summary statistics generation completed.")
        return summary_statistics