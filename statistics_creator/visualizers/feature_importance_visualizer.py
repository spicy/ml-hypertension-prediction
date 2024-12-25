import os

import matplotlib.pyplot as plt
import seaborn as sns
from config import feature_importance_config as config
from logger import log_execution_time, logger

from .base_visualizer import BaseVisualizer


class FeatureImportanceVisualizer(BaseVisualizer):
    """
    A visualizer for feature importance analysis results.
    Creates separate plots for Random Forest, Gradient Boosting, and F-score importance.
    """

    @log_execution_time
    def visualize(self, importance_data: dict, output_path: str) -> None:
        """
        Create and save feature importance visualizations.
        """
        logger.info("Creating feature importance visualizations...")

        # Plot Random Forest importance
        self._plot_importance(
            importance_data["rf_importance"],
            "Random Forest Feature Importance",
            os.path.join(output_path, config.RF_IMPORTANCE_PLOT),
        )

        # Plot Gradient Boosting importance
        self._plot_importance(
            importance_data["gb_importance"],
            "Gradient Boosting Feature Importance",
            os.path.join(output_path, config.GB_IMPORTANCE_PLOT),
        )

        # Plot F-score importance
        self._plot_importance(
            importance_data["f_score_importance"],
            "F-Score Feature Importance",
            os.path.join(output_path, config.F_SCORE_PLOT),
        )

        logger.info("Feature importance visualizations completed")

    def _plot_importance(self, importance_series, title, output_path):
        """Helper method to create importance plots"""
        # Get top 20 features
        importance_series = importance_series.head(20)

        plt.figure(figsize=(config.WIDTH, config.HEIGHT))

        sns.barplot(x=importance_series.values, y=importance_series.index, orient="h")

        plt.title(title, fontsize=config.TITLE_FONT_SIZE, pad=config.TITLE_PAD)
        plt.xlabel("Importance Score", fontsize=config.LABEL_FONT_SIZE)
        plt.ylabel("Features", fontsize=config.LABEL_FONT_SIZE)
        plt.xticks(fontsize=config.TICK_FONT_SIZE)
        plt.yticks(fontsize=config.TICK_FONT_SIZE)

        plt.tight_layout(pad=config.TIGHT_LAYOUT_PAD)

        try:
            plt.savefig(output_path, dpi=config.DPI, bbox_inches="tight")
            logger.info(f"Saved importance plot to: {output_path}")
        except IOError as e:
            logger.error(f"Error saving importance plot: {e}")
            raise
        finally:
            plt.close()
