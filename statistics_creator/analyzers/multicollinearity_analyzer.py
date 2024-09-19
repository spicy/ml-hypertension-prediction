import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from .base_analyzer import BaseAnalyzer
from logger import logger, log_execution_time

class MulticollinearityAnalyzer(BaseAnalyzer):
    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            logger.warning("No numeric columns found for multicollinearity analysis.")
            return pd.DataFrame()

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)

        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["Feature"] = numeric_df.columns
        vif_data["VIF"] = [self._calculate_vif(X_scaled, i) for i in range(X_scaled.shape[1])]

        # Sort by VIF in descending order
        vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

        logger.info("Multicollinearity analysis completed.")
        return vif_data

    def _calculate_vif(self, X, idx):
        y = X[:, idx]
        X_i = np.delete(X, idx, axis=1)
        r_squared = LinearRegression().fit(X_i, y).score(X_i, y)
        return 1 / (1 - r_squared)