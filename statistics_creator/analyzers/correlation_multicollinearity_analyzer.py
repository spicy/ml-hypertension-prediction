import numpy as np
import pandas as pd
from config import correlation_multicollinearity_config as config
from logger import log_execution_time, logger
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from .base_analyzer import BaseAnalyzer


class CorrelationMulticollinearityAnalyzer(BaseAnalyzer):
    """
    A class for analyzing correlations and multicollinearity in a DataFrame.

    This analyzer calculates the correlation matrix and Variance Inflation Factor (VIF)
    for numeric variables in the input DataFrame.
    """

    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Analyze the input DataFrame for correlations and multicollinearity.

        It performs the following steps:
        1. Selects numeric columns from the input DataFrame.
        2. Calculates the correlation matrix for these numeric columns.
        3. Imputes missing values and scales features for VIF calculation.
        4. Calculates VIF for each numeric feature.
        """
        logger.info("Starting correlation and multicollinearity analysis...")

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Calculate correlation matrix
        correlation_matrix = numeric_df.corr()

        # Impute missing values and scale features (for VIF calculation)
        imputer = SimpleImputer(strategy=config.IMPUTER_STRATEGY)
        X_imputed = imputer.fit_transform(numeric_df)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Calculate VIF
        vif_data = pd.DataFrame()
        vif_data["Feature"] = numeric_df.columns
        vif_data["VIF"] = [
            self._calculate_vif(X_scaled, i) for i in range(X_scaled.shape[1])
        ]
        vif_data = vif_data.sort_values(
            "VIF", ascending=config.VIF_SORT_ASCENDING
        ).reset_index(drop=True)

        logger.info("Correlation and multicollinearity analysis completed.")
        return {"correlation_matrix": correlation_matrix, "vif_data": vif_data}

    def _calculate_vif(self, X: np.ndarray, idx: int) -> float:
        """
        Calculate the Variance Inflation Factor for a given feature.

        This method uses linear regression to calculate the VIF value for a specified feature.
        If perfect multicollinearity is detected (R² = 1), returns infinity.
        """
        try:
            y = X[:, idx]
            X_without = np.delete(X, idx, axis=1)

            # Handle the case where there's only one feature left
            if X_without.size == 0:
                return 1.0

            r2 = r2_score(y, LinearRegression().fit(X_without, y).predict(X_without))

            # Handle perfect multicollinearity
            if np.isclose(r2, 1.0):
                return float("inf")

            return 1 / (1 - r2)
        except Exception as e:
            logger.warning(f"Error calculating VIF: {str(e)}. Returning infinity.")
            return float("inf")
