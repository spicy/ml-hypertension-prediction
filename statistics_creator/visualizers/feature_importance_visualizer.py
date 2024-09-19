import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .base_visualizer import BaseVisualizer
from config import feature_importance_config as config
from logger import logger, log_execution_time

class FeatureImportanceVisualizer(BaseVisualizer):
    """A visualizer for feature importance."""

    @log_execution_time
    def visualize(self, feature_importances: pd.Series, output_path: str) -> None:
        """
        Visualize the feature importances as a horizontal bar plot.

        Args:
            feature_importances (pd.Series): The feature importances to visualize.
            output_path (str): The path to save the visualization.
        """
        logger.info("Plotting feature importances...")
        plt.figure(figsize=(config.WIDTH, config.HEIGHT))

        # Plot horizontal bar chart
        sns.barplot(x=feature_importances.values, y=feature_importances.index, orient='h')
        plt.title('Feature Importance', fontsize=config.TITLE_FONT_SIZE, pad=config.TITLE_PAD)
        plt.xlabel('Importance', fontsize=config.LABEL_FONT_SIZE)
        plt.ylabel('Features', fontsize=config.LABEL_FONT_SIZE)
        plt.xticks(fontsize=config.TICK_FONT_SIZE)
        plt.yticks(fontsize=config.TICK_FONT_SIZE)

        # Adjust layout to prevent cutting off labels
        plt.tight_layout(pad=config.TIGHT_LAYOUT_PAD)

        # Save the plot
        png_path = os.path.join(output_path, 'feature_importance.png')
        plt.savefig(png_path, dpi=config.DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature importance plot saved to: {png_path}")