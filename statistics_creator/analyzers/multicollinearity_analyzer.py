import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from .base_analyzer import BaseAnalyzer
from logger import logger, log_execution_time

class MulticollinearityAnalyzer(BaseAnalyzer):
    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting multicollinearity analysis...")

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Impute missing values with mean
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(numeric_df)

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["Feature"] = numeric_df.columns
        vif_data["VIF"] = [self._calculate_vif(X_scaled, i) for i in range(X_scaled.shape[1])]

        # Sort by VIF value
        vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

        logger.info("Multicollinearity analysis completed.")
        return vif_data

    def _calculate_vif(self, X, idx):
        y = X[:, idx]
        X_without = np.delete(X, idx, axis=1)
        r2 = r2_score(y, LinearRegression().fit(X_without, y).predict(X_without))
        return 1 / (1 - r2)