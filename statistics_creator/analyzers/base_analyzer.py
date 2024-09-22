from abc import ABC, abstractmethod
import pandas as pd
from typing import Any
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

        Args:
            df (pd.DataFrame): The input DataFrame to analyze.

        Returns:
            Any: The result of the analysis.
        """
        pass

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input DataFrame by removing rows with missing values.

        Args:
            df (pd.DataFrame): The input DataFrame to preprocess.

        Returns:
            pd.DataFrame: The preprocessed DataFrame with NaN values removed.
        """
        return df.dropna()