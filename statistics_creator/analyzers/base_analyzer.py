from abc import ABC, abstractmethod
import pandas as pd
from typing import Any
from logger import log_execution_time

class BaseAnalyzer(ABC):
    @abstractmethod
    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> Any:
        pass

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()