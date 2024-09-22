import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .base_visualizer import BaseVisualizer
from config import summary_statistics_config as config
from logger import logger, log_execution_time

class SummaryStatisticsVisualizer(BaseVisualizer):
    """
    A visualizer that creates and saves a heatmap visualization
    of summary statistics for a given dataset.
    """

    @log_execution_time
    def visualize(self, summary_statistics: pd.DataFrame, output_path: str) -> None:
        """
        Visualize the summary statistics as a heatmap and save it to a file.

        Args:
            summary_statistics (pd.DataFrame): The summary statistics to visualize.
                This should be a DataFrame containing the statistics for each variable.
            output_path (str): The directory path where the visualization will be saved.

        Returns:
            None

        Raises:
            IOError: If there's an error saving the plot to the specified path.
        """
        logger.info("Plotting summary statistics...")
        plt.figure(figsize=(config.WIDTH, config.HEIGHT))

        sns.heatmap(summary_statistics, annot=True, fmt='.2f', cmap='YlGnBu')
        plt.title('Summary Statistics Heatmap', fontsize=config.TITLE_FONT_SIZE, pad=config.TITLE_PAD)
        plt.xticks(rotation=45, ha='right', fontsize=config.X_TICK_FONT_SIZE)
        plt.yticks(fontsize=config.Y_TICK_FONT_SIZE)

        plt.tight_layout(pad=config.TIGHT_LAYOUT_PAD)

        png_path = os.path.join(output_path, 'summary_statistics_heatmap.png')
        try:
            plt.savefig(png_path, dpi=config.DPI)
            logger.info(f"Summary statistics plot saved to: {png_path}")
        except IOError as e:
            logger.error(f"Error saving summary statistics plot: {e}")
            raise
        finally:
            plt.close()