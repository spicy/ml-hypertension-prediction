from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from logger import log_execution_time


class BaseAnalyzer(ABC):
    """
    Abstract base class for data analyzers.
    """

    @abstractmethod
    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> Any:
        """
        Perform analysis on the given DataFrame.
        """
        pass

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input DataFrame by removing rows with missing values.
        """
        return df.dropna()
