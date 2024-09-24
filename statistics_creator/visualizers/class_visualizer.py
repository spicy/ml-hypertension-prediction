import os
import pandas as pd
import matplotlib.pyplot as plt
from .base_visualizer import BaseVisualizer
from config import class_distribution_config as config
from logger import logger, log_execution_time

class ClassVisualizer(BaseVisualizer):
    """
    A visualizer for class distributions and balance.
    """

    PLOT_FILENAME = 'class_analysis.png'
    SUBPLOT_LAYOUT = (1, 2)
    PIE_CHART_POSITION = (1, 1)
    INFO_TEXT_POSITION = (1, 2)
    PIE_CHART_AUTOPCT = '%1.1f%%'
    PIE_CHART_START_ANGLE = 90
    TEXT_BOX_POSITION = (0.5, 0.5)
    TEXT_BOX_ALIGNMENT = ('center', 'center')
    TEXT_BOX_FACECOLOR = 'white'
    TEXT_BOX_ALPHA = 0.5

    @log_execution_time
    def visualize(self, class_analysis: pd.Series, output_path: str) -> None:
        """
        Visualize the class distribution as a pie chart and display class balance information.

        It creates a figure with two subplots:
        1. A pie chart showing the distribution of classes.
        2. A text box displaying information about class balance.

        Args:
            class_analysis (pd.Series): The class analysis results to visualize.
                Expected to contain keys:
                - 'distribution': A Series with class names as index and counts as values.
                - 'majority_class': The name of the majority class.
                - 'minority_class': The name of the minority class.
                - 'imbalance_ratio': The ratio of majority to minority class.
            output_path (str): The directory path where the visualization will be saved.

        Returns:
            None

        Raises:
            IOError: If there's an error saving the plot to the specified path.
        """
        logger.info("Plotting class distribution and balance...")
        plt.figure(figsize=(config.WIDTH, config.HEIGHT))

        # Pie chart for distribution
        plt.subplot(*self.SUBPLOT_LAYOUT, self.PIE_CHART_POSITION)
        distribution = class_analysis['distribution']
        plt.pie(distribution.values, labels=distribution.index, autopct=self.PIE_CHART_AUTOPCT, 
                startangle=self.PIE_CHART_START_ANGLE, textprops={'fontsize': config.PIE_TEXT_FONT_SIZE})
        plt.title('Class Distribution', fontsize=config.TITLE_FONT_SIZE, pad=config.TITLE_PAD)

        # Text information for class balance
        plt.subplot(*self.SUBPLOT_LAYOUT, self.INFO_TEXT_POSITION)
        plt.axis('off')
        info_text = (
            f"Majority Class: {class_analysis['majority_class']}\n"
            f"Minority Class: {class_analysis['minority_class']}\n"
            f"Imbalance Ratio: {class_analysis['imbalance_ratio']:.2f}"
        )
        plt.text(*self.TEXT_BOX_POSITION, info_text, ha=self.TEXT_BOX_ALIGNMENT[0], 
                 va=self.TEXT_BOX_ALIGNMENT[1], fontsize=config.TEXT_FONT_SIZE,
                 bbox=dict(facecolor=self.TEXT_BOX_FACECOLOR, alpha=self.TEXT_BOX_ALPHA))
        plt.title('Class Balance Information', fontsize=config.TITLE_FONT_SIZE, pad=config.TITLE_PAD)

        plt.tight_layout(pad=config.TIGHT_LAYOUT_PAD)

        png_path = os.path.join(output_path, self.PLOT_FILENAME)
        try:
            plt.savefig(png_path, dpi=config.DPI)
            logger.info(f"Class analysis plot saved to: {png_path}")
        except IOError as e:
            logger.error(f"Error saving class analysis plot: {e}")
            raise
        finally:
            plt.close()