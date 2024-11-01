import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import comprehensive_numerical_config as config
from logger import log_execution_time, logger

from .base_visualizer import BaseVisualizer


class ComprehensiveNumericalVisualizer(BaseVisualizer):
    """
    A visualizer class for creating and saving comprehensive numerical analysis plots,
    including distribution plots and summary statistics heatmap.
    """

    @log_execution_time
    def visualize(self, data: dict, output_path: str) -> None:
        """
        Visualize the comprehensive numerical analysis results and save the plots.
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
            ax1.set_title(f"Distribution of {column}", fontsize=config.TITLE_FONT_SIZE)
            ax1.set_xlabel("")
            ax1.set_ylabel("Frequency", fontsize=config.LABEL_FONT_SIZE)

            # Box plot
            ax2 = plt.subplot(2, 1, 2)
            sns.boxplot(x=stats, ax=ax2, width=config.BOXPLOT_WIDTH)
            ax2.set_xlabel("Value", fontsize=config.LABEL_FONT_SIZE)

            plt.tight_layout(pad=config.TIGHT_LAYOUT_PAD)

            png_path = os.path.join(
                output_path, config.DISTRIBUTION_PLOT_FILENAME.format(column)
            )
            plt.savefig(png_path, dpi=config.DPI)
            plt.close()

        logger.info(f"Numerical distribution plots saved to: {output_path}")

    def _plot_summary_heatmap(self, data: dict, output_path: str) -> None:
        summary_df = pd.DataFrame(data).T

        plt.figure(figsize=(config.WIDTH, config.HEIGHT))
        sns.heatmap(
            summary_df,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            annot_kws={"size": config.HEATMAP_ANNOT_SIZE},
        )
        plt.title(
            "Summary Statistics Heatmap",
            fontsize=config.TITLE_FONT_SIZE,
            pad=config.TITLE_PAD,
        )
        plt.xticks(rotation=45, ha="right", fontsize=config.X_TICK_FONT_SIZE)
        plt.yticks(fontsize=config.Y_TICK_FONT_SIZE)

        plt.tight_layout(pad=config.TIGHT_LAYOUT_PAD)

        png_path = os.path.join(output_path, config.SUMMARY_HEATMAP_FILENAME)
        plt.savefig(png_path, dpi=config.DPI)
        plt.close()

        logger.info(f"Summary statistics heatmap saved to: {png_path}")
