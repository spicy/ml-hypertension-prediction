import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .base_visualizer import BaseVisualizer
from config import comprehensive_numerical_config as config
from logger import logger, log_execution_time

class ComprehensiveNumericalVisualizer(BaseVisualizer):
    """
    A visualizer class for creating and saving comprehensive numerical analysis plots,
    including distribution plots and summary statistics heatmap.
    """

    @log_execution_time
    def visualize(self, data: dict, output_path: str) -> None:
        """
        Visualize the comprehensive numerical analysis results and save the plots.

        Args:
            data (dict): A dictionary containing numerical analysis results for each column.
            output_path (str): The directory path where the visualization plots will be saved.

        Returns:
            None

        Raises:
            IOError: If there's an error saving the plots to the specified path.
        """
        logger.info("Plotting comprehensive numerical analysis...")

        self._plot_distributions(data, output_path)
        self._plot_summary_heatmap(data, output_path)

    def _plot_distributions(self, data: dict, output_path: str) -> None:
        for column, stats in data.items():
            plt.figure(figsize=(config.WIDTH, config.HEIGHT))

            # Histogram and KDE
            ax1 = plt.subplot(2, 1, 1)
            sns.histplot(stats, kde=True, ax=ax1, bins=config.HIST_BINS)
            ax1.set_title(f'Distribution of {column}', fontsize=config.TITLE_FONT_SIZE)
            ax1.set_xlabel('')
            ax1.set_ylabel('Frequency', fontsize=config.LABEL_FONT_SIZE)

            # Box plot
            ax2 = plt.subplot(2, 1, 2)
            sns.boxplot(x=stats, ax=ax2, width=config.BOXPLOT_WIDTH)
            ax2.set_xlabel('Value', fontsize=config.LABEL_FONT_SIZE)

            plt.tight_layout(pad=config.TIGHT_LAYOUT_PAD)

            png_path = os.path.join(output_path, f'numerical_distribution_{column}.png')
            plt.savefig(png_path, dpi=config.DPI)
            plt.close()

        logger.info(f"Numerical distribution plots saved to: {output_path}")

    def _plot_summary_heatmap(self, data: dict, output_path: str) -> None:
        summary_df = pd.DataFrame(data).T

        plt.figure(figsize=(config.WIDTH, config.HEIGHT))
        sns.heatmap(summary_df, annot=True, fmt='.2f', cmap='YlGnBu', 
                    annot_kws={"size": config.HEATMAP_ANNOT_SIZE})
        plt.title('Summary Statistics Heatmap', fontsize=config.TITLE_FONT_SIZE, pad=config.TITLE_PAD)
        plt.xticks(rotation=45, ha='right', fontsize=config.X_TICK_FONT_SIZE)
        plt.yticks(fontsize=config.Y_TICK_FONT_SIZE)

        plt.tight_layout(pad=config.TIGHT_LAYOUT_PAD)

        png_path = os.path.join(output_path, 'summary_statistics_heatmap.png')
        plt.savefig(png_path, dpi=config.DPI)
        plt.close()

        logger.info(f"Summary statistics heatmap saved to: {png_path}")