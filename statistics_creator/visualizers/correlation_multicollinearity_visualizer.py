import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import correlation_multicollinearity_config as config
from logger import log_execution_time, logger

from .base_visualizer import BaseVisualizer


class CorrelationMulticollinearityVisualizer(BaseVisualizer):
    """
    A visualizer for creating and saving correlation matrix heatmaps and multicollinearity plots.
    """

    @log_execution_time
    def visualize(self, data: dict, output_path: str) -> None:
        """
        Visualize the correlation matrix as a heatmap and multicollinearity using VIF.
        """
        self._visualize_correlation(data["correlation_matrix"], output_path)
        self._visualize_multicollinearity(data["vif_data"], output_path)

    def _visualize_correlation(
        self, correlation_matrix: pd.DataFrame, output_path: str
    ) -> None:
        logger.info("Plotting correlation matrix...")
        plt.figure(figsize=(config.CORR_WIDTH, config.CORR_HEIGHT))

        sns.heatmap(
            correlation_matrix,
            cmap="RdBu",
            vmin=-1,
            vmax=1,
            center=0,
            annot=True,
            fmt=".2f",
            square=True,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": config.CORR_ANNOT_FONT_SIZE},
        )

        plt.title(
            "Correlation Matrix Heatmap (Top 20 Features)",
            fontsize=config.TITLE_FONT_SIZE,
            pad=config.TITLE_PAD,
        )
        plt.xticks(rotation=90, fontsize=config.CORR_X_TICK_FONT_SIZE)
        plt.yticks(rotation=0, fontsize=config.CORR_Y_TICK_FONT_SIZE)

        plt.tight_layout(pad=config.CORR_TIGHT_LAYOUT_PAD)

        png_path = os.path.join(output_path, config.CORRELATION_PLOT_FILENAME)
        plt.savefig(png_path, dpi=config.DPI)
        plt.close()
        logger.info(f"Correlation matrix plot saved to: {png_path}")

    def _visualize_multicollinearity(
        self, vif_data: pd.DataFrame, output_path: str
    ) -> None:
        if vif_data.empty:
            logger.warning("No data to visualize for multicollinearity.")
            return

        logger.info("Plotting multicollinearity (VIF) data...")
        plt.figure(figsize=(config.VIF_WIDTH, config.VIF_HEIGHT))

        sns.barplot(x="VIF", y="Feature", data=vif_data)
        plt.title(
            "Variance Inflation Factor (VIF) for Top 20 Features",
            fontsize=config.TITLE_FONT_SIZE,
        )
        plt.xlabel("VIF", fontsize=config.VIF_LABEL_FONT_SIZE)
        plt.ylabel("Features", fontsize=config.VIF_LABEL_FONT_SIZE)
        plt.xticks(fontsize=config.VIF_TICK_FONT_SIZE)
        plt.yticks(fontsize=config.VIF_TICK_FONT_SIZE)

        plt.axvline(
            x=config.VIF_THRESHOLD,
            color="r",
            linestyle="--",
            label=f"VIF Threshold ({config.VIF_THRESHOLD})",
        )
        plt.legend(fontsize=config.VIF_LEGEND_FONT_SIZE)

        plt.tight_layout()

        png_path = os.path.join(output_path, config.MULTICOLLINEARITY_PLOT_FILENAME)
        try:
            plt.savefig(png_path, dpi=config.DPI)
            logger.info(f"Multicollinearity (VIF) plot saved to: {png_path}")
        except IOError as e:
            logger.error(f"Error saving multicollinearity plot: {e}")
            raise
        finally:
            plt.close()
