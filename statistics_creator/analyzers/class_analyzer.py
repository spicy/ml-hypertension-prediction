import pandas as pd
from .base_analyzer import BaseAnalyzer
from config import class_distribution_config as config
from logger import logger, log_execution_time

class ClassAnalyzer(BaseAnalyzer):
    """
    A class for analyzing the distribution of classes in a target column of a DataFrame.
    """

    def __init__(self, target_column: str):
        """
        Initialize the ClassAnalyzer.

        Args:
            target_column (str): The name of the target column to analyze.
        """
        self.target_column = target_column

    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> pd.Series:
        """
        Analyze the class distribution of the target column in the input DataFrame.

        It performs the following steps:
        1. Checks if the target column exists in the DataFrame.
        2. Calculates the distribution of classes in the target column.
        3. Identifies the majority and minority classes.
        4. Calculates the imbalance ratio between the majority and minority classes.

        Args:
            df (pd.DataFrame): The input DataFrame to analyze.

        Returns:
            pd.Series: A Series containing the following information:
                - distribution: Percentage distribution of classes
                - majority_class: The class with the highest frequency
                - minority_class: The class with the lowest frequency
                - imbalance_ratio: Ratio of majority class count to minority class count

        Raises:
            ValueError: If the target column is not found in the DataFrame.
        """
        if self.target_column not in df.columns:
            logger.error(f"Target column '{self.target_column}' not found in the dataset.")
            raise ValueError(f"Target column '{self.target_column}' not found in the dataset.")

        target_values = df[self.target_column].dropna()
        value_counts = target_values.value_counts()
        total = len(target_values)

        distribution = (value_counts / total) * 100

        majority_class = value_counts.index[0]
        minority_class = value_counts.index[-1]
        imbalance_ratio = value_counts[majority_class] / value_counts[minority_class]

        logger.info(f"Class analysis completed for column '{self.target_column}'")
        return pd.Series({
            config.DISTRIBUTION_KEY: distribution,
            config.MAJORITY_CLASS_KEY: majority_class,
            config.MINORITY_CLASS_KEY: minority_class,
            config.IMBALANCE_RATIO_KEY: imbalance_ratio
        })