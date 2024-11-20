import os

import matplotlib.pyplot as plt
import pandas as pd
from config import class_distribution_config as config
from logger import log_execution_time, logger

from .base_visualizer import BaseVisualizer


class ClassVisualizer(BaseVisualizer):
    """
    A visualizer for class distributions and balance.
    """

    @log_execution_time
    def visualize(self, class_analysis: pd.Series, output_path: str) -> None:
        """
        Visualize the class distribution as a pie chart and display class balance information.

        It creates a figure with two subplots:
        1. A pie chart showing the distribution of classes.
        2. A text box displaying information about class balance.
        """
        logger.info("Plotting class distribution and balance...")
        plt.figure(figsize=(config.WIDTH, config.HEIGHT))

        # Pie chart for distribution
        plt.subplot(*config.SUBPLOT_LAYOUT, config.PIE_CHART_POSITION)
        distribution = class_analysis["distribution"]
        plt.pie(
            distribution.values,
            labels=distribution.index,
            autopct=config.PIE_CHART_AUTOPCT,
            startangle=config.PIE_CHART_START_ANGLE,
            textprops={"fontsize": config.PIE_TEXT_FONT_SIZE},
        )
        plt.title(
            "Class Distribution", fontsize=config.TITLE_FONT_SIZE, pad=config.TITLE_PAD
        )

        # Text information for class balance
        plt.subplot(*config.SUBPLOT_LAYOUT, config.INFO_TEXT_POSITION)
        plt.axis("off")
        info_text = (
            f"Majority Class: {class_analysis['majority_class']}\n"
            f"Minority Class: {class_analysis['minority_class']}\n"
            f"Imbalance Ratio: {class_analysis['imbalance_ratio']:.2f}"
        )
        plt.text(
            *config.TEXT_BOX_POSITION,
            info_text,
            ha=config.TEXT_BOX_ALIGNMENT[0],
            va=config.TEXT_BOX_ALIGNMENT[1],
            fontsize=config.TEXT_FONT_SIZE,
            bbox=dict(facecolor=config.TEXT_BOX_FACECOLOR, alpha=config.TEXT_BOX_ALPHA),
        )

        plt.tight_layout(pad=config.TIGHT_LAYOUT_PAD)

        png_path = os.path.join(output_path, config.PLOT_FILENAME)
        try:
            plt.savefig(png_path, dpi=config.DPI)
            logger.info(f"Class analysis plot saved to: {png_path}")
        except IOError as e:
            logger.error(f"Error saving class analysis plot: {e}")
            raise
        finally:
            plt.close()
