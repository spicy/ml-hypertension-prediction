import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .base_visualizer import BaseVisualizer
from config import summary_statistics_config as config
from logger import logger, log_execution_time

class SummaryStatisticsVisualizer(BaseVisualizer):
    """A visualizer for summary statistics."""

    @log_execution_time
    def visualize(self, summary_statistics: pd.DataFrame, output_path: str) -> None:
        """
        Visualize the summary statistics as a heatmap.

        Args:
            summary_statistics (pd.DataFrame): The summary statistics to visualize.
            output_path (str): The path to save the visualization.
        """
        logger.info("Plotting summary statistics...")
        plt.figure(figsize=(config.WIDTH, config.HEIGHT))

        sns.heatmap(summary_statistics, annot=True, fmt='.2f', cmap='YlGnBu')
        plt.title('Summary Statistics Heatmap', fontsize=config.TITLE_FONT_SIZE, pad=config.TITLE_PAD)
        plt.xticks(rotation=45, ha='right', fontsize=config.X_TICK_FONT_SIZE)
        plt.yticks(fontsize=config.Y_TICK_FONT_SIZE)

        plt.tight_layout(pad=config.TIGHT_LAYOUT_PAD)

        png_path = os.path.join(output_path, 'summary_statistics_heatmap.png')
        plt.savefig(png_path, dpi=config.DPI)
        plt.close()
        logger.info(f"Summary statistics plot saved to: {png_path}")