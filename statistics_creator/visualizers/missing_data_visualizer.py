import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .base_visualizer import BaseVisualizer
from config import missing_data_config as config
from logger import logger, log_execution_time

class MissingDataVisualizer(BaseVisualizer):
    """
    A visualizer for creating and saving missing data percentage plots.
    """

    PLOT_TITLE = 'Percentage of Missing Data by Column'
    X_LABEL = 'Columns'
    Y_LABEL = 'Percentage of Missing Data'
    PLOT_FILENAME = 'missing_data_percentage.png'

    @log_execution_time
    def visualize(self, missing_percentage_sorted: pd.Series, output_path: str) -> None:
        """
        Visualize the percentage of missing data for each column and save the plot.

        Args:
            missing_percentage_sorted (pd.Series): Sorted series of missing data percentages.
                The index contains column names and values are the corresponding
                percentages of missing data.
            output_path (str): The directory path where the visualization will be saved.

        Returns:
            None

        Raises:
            IOError: If there's an error saving the plot to the specified path.
        """
        logger.info("Plotting missing data...")
        plt.figure(figsize=(config.WIDTH, config.HEIGHT))
        ax = sns.barplot(x=missing_percentage_sorted.index, y=missing_percentage_sorted.values)

        plt.title(self.PLOT_TITLE, fontsize=config.TITLE_FONT_SIZE, pad=config.TITLE_PAD)
        plt.xlabel(self.X_LABEL, fontsize=config.X_LABEL_FONT_SIZE, labelpad=config.LABEL_PAD)
        plt.ylabel(self.Y_LABEL, fontsize=config.Y_LABEL_FONT_SIZE, labelpad=config.LABEL_PAD)
        plt.xticks(rotation=90, fontsize=config.X_TICK_FONT_SIZE)
        plt.yticks(fontsize=config.Y_TICK_FONT_SIZE)

        for i, v in enumerate(missing_percentage_sorted.values):
            ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=config.TEXT_FONT_SIZE)

        plt.ylim(0, max(missing_percentage_sorted) * config.YLIM_MULTIPLIER)
        plt.tight_layout(pad=config.TIGHT_LAYOUT_PAD)

        png_path = os.path.join(output_path, self.PLOT_FILENAME)
        try:
            plt.savefig(png_path, dpi=config.DPI)
            logger.info(f"Missing data plot saved to: {png_path}")
        except IOError as e:
            logger.error(f"Error saving missing data plot: {e}")
            raise
        finally:
            plt.close()