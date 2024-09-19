import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .base_visualizer import BaseVisualizer
from config import correlation_config as config
from logger import logger, log_execution_time

class CorrelationVisualizer(BaseVisualizer):
    """A visualizer for correlation matrices."""

    @log_execution_time
    def visualize(self, correlation_matrix: pd.DataFrame, output_path: str) -> None:
        """
        Visualize the correlation matrix as a heatmap.

        Args:
            correlation_matrix (pd.DataFrame): The correlation matrix to visualize.
            output_path (str): The path to save the visualization.
        """
        logger.info("Plotting correlation matrix...")
        plt.figure(figsize=(config.WIDTH, config.HEIGHT))

        sns.heatmap(correlation_matrix,
                    cmap='RdBu',
                    vmin=-1,
                    vmax=1,
                    center=0,
                    annot=True,
                    fmt='.2f',
                    square=True,
                    cbar_kws={"shrink": .8},
                    annot_kws={"size": config.ANNOT_FONT_SIZE})

        plt.title('Correlation Matrix Heatmap', fontsize=config.TITLE_FONT_SIZE, pad=config.TITLE_PAD)
        plt.xticks(rotation=90, fontsize=config.X_TICK_FONT_SIZE)
        plt.yticks(rotation=0, fontsize=config.Y_TICK_FONT_SIZE)

        plt.tight_layout(pad=config.TIGHT_LAYOUT_PAD)

        png_path = os.path.join(output_path, 'correlation_matrix_heatmap.png')
        plt.savefig(png_path, dpi=config.DPI)
        plt.close()
        logger.info(f"Correlation matrix plot saved to: {png_path}")