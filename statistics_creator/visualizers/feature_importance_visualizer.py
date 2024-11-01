import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import feature_importance_config as config
from logger import log_execution_time, logger

from .base_visualizer import BaseVisualizer


class FeatureImportanceVisualizer(BaseVisualizer):
    """
    A visualizer for creating and saving feature importance plots.
    """

    @log_execution_time
    def visualize(self, feature_importances: pd.Series, output_path: str) -> None:
        """
        Visualize the feature importances as a horizontal bar plot and save it.
        """
        logger.info("Plotting feature importances...")
        plt.figure(figsize=(config.WIDTH, config.HEIGHT))

        # Plot horizontal bar chart
        sns.barplot(
            x=feature_importances.values, y=feature_importances.index, orient="h"
        )
        plt.title(
            config.PLOT_TITLE, fontsize=config.TITLE_FONT_SIZE, pad=config.TITLE_PAD
        )
        plt.xlabel(config.X_LABEL, fontsize=config.LABEL_FONT_SIZE)
        plt.ylabel(config.Y_LABEL, fontsize=config.LABEL_FONT_SIZE)
        plt.xticks(fontsize=config.TICK_FONT_SIZE)
        plt.yticks(fontsize=config.TICK_FONT_SIZE)

        # Adjust layout to prevent cutting off labels
        plt.tight_layout(pad=config.TIGHT_LAYOUT_PAD)

        # Save the plot
        png_path = os.path.join(output_path, config.PLOT_FILENAME)
        try:
            plt.savefig(png_path, dpi=config.DPI, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to: {png_path}")
        except IOError as e:
            logger.error(f"Error saving feature importance plot: {e}")
            raise
        finally:
            plt.close()
