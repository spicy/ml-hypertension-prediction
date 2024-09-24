import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .base_visualizer import BaseVisualizer
from config import outlier_config as config
from logger import logger, log_execution_time

class OutlierVisualizer(BaseVisualizer):
    """
    A visualizer class for creating and saving outlier plots.
    """

    @log_execution_time
    def visualize(self, outlier_data: dict, output_path: str) -> None:
        """
        Visualize outliers for each column in the dataset and save the plots.

        Args:
            outlier_data (dict): A dictionary containing outlier information for each column.
                The keys are column names, and the values are dictionaries with 'outliers',
                'lower_bound', and 'upper_bound' keys.
            output_path (str): The directory path where the visualization plots will be saved.

        Returns:
            None

        Raises:
            IOError: If there's an error saving the plot to the specified path.
        """
        logger.info("Plotting outlier data...")

        for column, data in outlier_data.items():
            plt.figure(figsize=(config.WIDTH, config.HEIGHT))

            sns.boxplot(x=column, data=pd.DataFrame({column: data['outliers']}))
            plt.title(f'Outliers in {column}', fontsize=config.TITLE_FONT_SIZE)
            plt.xlabel(column, fontsize=config.LABEL_FONT_SIZE)
            plt.ylabel('Value', fontsize=config.LABEL_FONT_SIZE)
            plt.xticks(fontsize=config.TICK_FONT_SIZE)
            plt.yticks(fontsize=config.TICK_FONT_SIZE)

            # Add lines for lower and upper bounds
            plt.axhline(y=data['lower_bound'], color=config.LOWER_BOUND_COLOR, linestyle=config.BOUND_LINESTYLE, label='Lower Bound')
            plt.axhline(y=data['upper_bound'], color=config.UPPER_BOUND_COLOR, linestyle=config.BOUND_LINESTYLE, label='Upper Bound')
            plt.legend(fontsize=config.LEGEND_FONT_SIZE)

            plt.tight_layout()

            # Save the plot
            png_path = os.path.join(output_path, f'{config.PLOT_FILE_PREFIX}{column}{config.PLOT_FILE_EXTENSION}')
            try:
                plt.savefig(png_path, dpi=config.DPI)
            except IOError as e:
                logger.error(f"Error saving outlier plot for {column}: {e}")
                raise
            finally:
                plt.close()

        logger.info(f"Outlier plots saved to: {output_path}")