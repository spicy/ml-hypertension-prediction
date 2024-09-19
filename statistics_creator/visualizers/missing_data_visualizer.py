import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .base_visualizer import BaseVisualizer
from config import missing_data_config as config
from logger import logger, log_execution_time

class MissingDataVisualizer(BaseVisualizer):
    """A visualizer for missing data."""

    @log_execution_time
    def visualize(self, missing_percentage_sorted: pd.Series, output_path: str) -> None:
        """
        Visualize the percentage of missing data for each column.

        Args:
            missing_percentage_sorted (pd.Series): Sorted series of missing data percentages.
            output_path (str): The path to save the visualization.
        """
        logger.info("Plotting missing data...")
        plt.figure(figsize=(config.WIDTH, config.HEIGHT))
        ax = sns.barplot(x=missing_percentage_sorted.index, y=missing_percentage_sorted.values)

        plt.title('Percentage of Missing Data by Column', fontsize=config.TITLE_FONT_SIZE, pad=config.TITLE_PAD)
        plt.xlabel('Columns', fontsize=config.X_LABEL_FONT_SIZE, labelpad=config.LABEL_PAD)
        plt.ylabel('Percentage of Missing Data', fontsize=config.Y_LABEL_FONT_SIZE, labelpad=config.LABEL_PAD)
        plt.xticks(rotation=90, fontsize=config.X_TICK_FONT_SIZE)
        plt.yticks(fontsize=config.Y_TICK_FONT_SIZE)

        for i, v in enumerate(missing_percentage_sorted.values):
            ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=config.TEXT_FONT_SIZE)

        plt.ylim(0, max(missing_percentage_sorted) * config.YLIM_MULTIPLIER)
        plt.tight_layout(pad=config.TIGHT_LAYOUT_PAD)

        png_path = os.path.join(output_path, 'missing_data_percentage.png')
        plt.savefig(png_path, dpi=config.DPI)
        plt.close()
        logger.info(f"Missing data plot saved to: {png_path}")