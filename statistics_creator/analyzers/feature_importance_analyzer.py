import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from .base_analyzer import BaseAnalyzer
from logger import logger, log_execution_time

class FeatureImportanceAnalyzer(BaseAnalyzer):
    def __init__(self, target_column: str):
        self.target_column = target_column

    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> pd.Series:
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
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # Get feature importances
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        importances_sorted = importances.sort_values(ascending=False)

        logger.info(f"Feature importance analysis completed. Rows removed due to NaN in target: {len(df) - len(df_clean)}")
        return importances_sorted