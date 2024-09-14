import os
import pandas as pd
import matplotlib.pyplot as plt
from .base_visualizer import BaseVisualizer
from config import class_distribution_config as config
from logger import logger, log_execution_time

class ClassDistributionVisualizer(BaseVisualizer):
    """A visualizer for class distributions."""

    @log_execution_time
    def visualize(self, class_distribution: pd.Series, output_path: str) -> None:
        """
        Visualize the class distribution as a pie chart.

        Args:
            class_distribution (pd.Series): The class distribution to visualize.
            output_path (str): The path to save the visualization.
        """
        logger.info("Plotting class distribution...")
        plt.figure(figsize=(config.WIDTH, config.HEIGHT))

        plt.pie(class_distribution.values, labels=class_distribution.index, autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': config.PIE_TEXT_FONT_SIZE})
        plt.title('Class Distribution', fontsize=config.TITLE_FONT_SIZE, pad=config.TITLE_PAD)

        plt.axis('equal')
        plt.tight_layout(pad=config.TIGHT_LAYOUT_PAD)

        png_path = os.path.join(output_path, 'class_distribution_pie.png')
        plt.savefig(png_path, dpi=config.DPI)
        plt.close()
        logger.info(f"Class distribution plot saved to: {png_path}")