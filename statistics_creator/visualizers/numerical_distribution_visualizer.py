import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .base_visualizer import BaseVisualizer
from config import numerical_distribution_config as config
from logger import logger, log_execution_time

class NumericalDistributionVisualizer(BaseVisualizer):
    """
    A visualizer class for creating and saving numerical distribution plots.
    """

    @log_execution_time
    def visualize(self, distribution_data: dict[str, dict[str, float]], output_path: str) -> None:
        """
        Visualize the numerical distribution of each column in the dataset and save the plots.

        Args:
            distribution_data (dict[str, dict[str, float]]): A dictionary containing distribution data for each column.
                The keys are column names, and the values are dictionaries with distribution statistics.
            output_path (str): The directory path where the visualization plots will be saved.

        Returns:
            None

        Raises:
            IOError: If there's an error saving the plot to the specified path.
        """
        logger.info("Plotting numerical distributions...")

        for column, stats in distribution_data.items():
            plt.figure(figsize=(config.WIDTH, config.HEIGHT))

            # Create a subplot for the histogram and KDE
            ax1 = plt.subplot(2, 1, 1)
            sns.histplot(data=pd.Series(stats), kde=True, ax=ax1)
            ax1.set_title(f'Distribution of {column}', fontsize=config.TITLE_FONT_SIZE)
            ax1.set_xlabel('')

            # Create a subplot for the box plot
            ax2 = plt.subplot(2, 1, 2)
            sns.boxplot(data=pd.Series(stats), ax=ax2)
            ax2.set_xlabel('Value', fontsize=config.LABEL_FONT_SIZE)

            plt.tight_layout(pad=config.TIGHT_LAYOUT_PAD)

            # Save the plot
            png_path = os.path.join(output_path, f'numerical_distribution_{column}.png')
            try:
                plt.savefig(png_path, dpi=config.DPI)
            except IOError as e:
                logger.error(f"Error saving numerical distribution plot for {column}: {e}")
                raise
            finally:
                plt.close()

        logger.info(f"Numerical distribution plots saved to: {output_path}")