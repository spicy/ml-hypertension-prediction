import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from .base_analyzer import BaseAnalyzer
from logger import logger, log_execution_time

class FeatureImportanceAnalyzer(BaseAnalyzer):
    """
    A class for analyzing feature importance in a DataFrame using Random Forest.
    """

    N_ESTIMATORS = 100
    RANDOM_STATE = 42

    def __init__(self, target_column: str):
        """
        Initialize the FeatureImportanceAnalyzer.

        Args:
            target_column (str): The name of the target column in the DataFrame.
        """
        self.target_column = target_column

    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> pd.Series:
        """
        Analyze the input DataFrame to determine feature importance.

        It performs the following steps:
        1. Removes rows with NaN values in the target column.
        2. Handles categorical variables using one-hot encoding.
        3. Encodes non-numeric target variables.
        4. Trains a Random Forest classifier.
        5. Calculates and returns feature importances.

        Args:
            df (pd.DataFrame): The input DataFrame to analyze.

        Returns:
            pd.Series: A Series containing feature importances, sorted in descending order.

        Raises:
            ValueError: If the target column is not found in the DataFrame.
        """
        if self.target_column not in df.columns:
            logger.error(f"Target column '{self.target_column}' not found in the dataset.")
            raise ValueError(f"Target column '{self.target_column}' not found in the dataset.")

        # Remove rows with NaN values in the target column
        df_clean = df.dropna(subset=[self.target_column])

        X = df_clean.drop(columns=[self.target_column])
        y = df_clean[self.target_column]

        # Handle categorical variables
        X = pd.get_dummies(X)

        # Handle non-numeric target variable
        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Train a Random Forest classifier
        rf = RandomForestClassifier(n_estimators=self.N_ESTIMATORS, random_state=self.RANDOM_STATE)
        rf.fit(X, y)

        # Get feature importances
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        importances_sorted = importances.sort_values(ascending=False)

        logger.info(f"Feature importance analysis completed. Rows removed due to NaN in target: {len(df) - len(df_clean)}")
        return importances_sorted