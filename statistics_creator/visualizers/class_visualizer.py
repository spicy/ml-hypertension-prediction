import os
import pandas as pd
import matplotlib.pyplot as plt
from .base_visualizer import BaseVisualizer
from config import class_distribution_config as config
from logger import logger, log_execution_time

class ClassVisualizer(BaseVisualizer):
    """A visualizer for class distributions and balance."""

    @log_execution_time
    def visualize(self, class_analysis: pd.Series, output_path: str) -> None:
        """
        Visualize the class distribution as a pie chart and display class balance information.

        Args:
            class_analysis (pd.Series): The class analysis results to visualize.
            output_path (str): The path to save the visualization.
        """
        logger.info("Plotting class distribution and balance...")
        plt.figure(figsize=(config.WIDTH, config.HEIGHT))

        # Pie chart for distribution
        plt.subplot(1, 2, 1)
        distribution = class_analysis['distribution']
        plt.pie(distribution.values, labels=distribution.index, autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': config.PIE_TEXT_FONT_SIZE})
        plt.title('Class Distribution', fontsize=config.TITLE_FONT_SIZE, pad=config.TITLE_PAD)

        # Text information for class balance
        plt.subplot(1, 2, 2)
        plt.axis('off')
        info_text = (
            f"Majority Class: {class_analysis['majority_class']}\n"
            f"Minority Class: {class_analysis['minority_class']}\n"
            f"Imbalance Ratio: {class_analysis['imbalance_ratio']:.2f}"
        )
        plt.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=config.TEXT_FONT_SIZE,
                 bbox=dict(facecolor='white', alpha=0.5))
        plt.title('Class Balance Information', fontsize=config.TITLE_FONT_SIZE, pad=config.TITLE_PAD)

        plt.tight_layout(pad=config.TIGHT_LAYOUT_PAD)

        png_path = os.path.join(output_path, 'class_analysis.png')
        plt.savefig(png_path, dpi=config.DPI)
        plt.close()
        logger.info(f"Class analysis plot saved to: {png_path}")