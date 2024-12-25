import numpy as np
import pandas as pd
from config import feature_importance_config as config
from logger import log_execution_time, logger
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder

from .base_analyzer import BaseAnalyzer


class FeatureImportanceAnalyzer(BaseAnalyzer):
    """
    A class for analyzing feature importance using multiple methods:
    1. Random Forest feature importance
    2. Gradient Boosting feature importance
    3. F-score based feature selection (SelectKBest)
    """

    def __init__(self, target_column: str):
        self.target_column = target_column
        self.k_best = config.K_BEST_FEATURES

    @log_execution_time
    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Analyze feature importance using multiple methods.

        Returns:
            dict containing:
            - rf_importance: Random Forest feature importance scores
            - gb_importance: Gradient Boosting feature importance scores
            - f_score_importance: F-score based feature importance
            - selected_features: Top K features selected by F-score
        """
        if self.target_column not in df.columns:
            logger.error(f"Target column '{self.target_column}' not found")
            raise ValueError(f"Target column '{self.target_column}' not found")

        # Drop index column if present (SEQN)
        if "SEQN" in df.columns:
            df = df.drop("SEQN", axis=1)

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Handle non-numeric target variable
        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Random Forest importance
        rf = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
            random_state=config.RANDOM_STATE,
        )
        rf.fit(X, y)
        rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
            ascending=False
        )

        # Gradient Boosting importance
        gb = GradientBoostingClassifier(
            n_estimators=config.GB_N_ESTIMATORS,
            learning_rate=config.GB_LEARNING_RATE,
            max_depth=config.GB_MAX_DEPTH,
            min_samples_leaf=config.GB_MIN_SAMPLES_LEAF,
            min_samples_split=config.GB_MIN_SAMPLES_SPLIT,
            random_state=config.RANDOM_STATE,
        )
        gb.fit(X, y)
        gb_importance = pd.Series(gb.feature_importances_, index=X.columns).sort_values(
            ascending=False
        )

        # F-score based feature selection
        selector = SelectKBest(f_classif, k=self.k_best)
        selector.fit(X, y)
        f_scores = pd.Series(selector.scores_, index=X.columns).sort_values(
            ascending=False
        )

        # Get selected feature mask and names
        selected_features = X.columns[selector.get_support()].tolist()

        # Limit all series to top 20 features
        rf_importance = rf_importance.head(20)
        gb_importance = gb_importance.head(20)
        f_scores = f_scores.head(20)
        selected_features = selected_features[:20]

        logger.info(
            f"Feature importance analysis completed. Selected {len(selected_features)} features."
        )
        return {
            "rf_importance": rf_importance,
            "gb_importance": gb_importance,
            "f_score_importance": f_scores,
            "selected_features": selected_features,
        }
