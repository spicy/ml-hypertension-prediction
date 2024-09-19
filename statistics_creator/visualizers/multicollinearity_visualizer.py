import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .base_visualizer import BaseVisualizer
from config import multicollinearity_config as config
from logger import logger, log_execution_time

class MulticollinearityVisualizer(BaseVisualizer):
    @log_execution_time
    def visualize(self, vif_data: pd.DataFrame, output_path: str) -> None:
        if vif_data.empty:
            logger.warning("No data to visualize for multicollinearity.")
            return

        logger.info("Plotting multicollinearity (VIF) data...")
        plt.figure(figsize=(config.WIDTH, config.HEIGHT))

        sns.barplot(x="VIF", y="Feature", data=vif_data)
        plt.title("Variance Inflation Factor (VIF) for Features", fontsize=config.TITLE_FONT_SIZE)
        plt.xlabel("VIF", fontsize=config.LABEL_FONT_SIZE)
        plt.ylabel("Features", fontsize=config.LABEL_FONT_SIZE)
        plt.xticks(fontsize=config.TICK_FONT_SIZE)
        plt.yticks(fontsize=config.TICK_FONT_SIZE)

        # Add VIF threshold line
        plt.axvline(x=config.VIF_THRESHOLD, color='r', linestyle='--', label=f'VIF Threshold ({config.VIF_THRESHOLD})')
        plt.legend(fontsize=config.LEGEND_FONT_SIZE)

        plt.tight_layout()

        # Save the plot
        png_path = os.path.join(output_path, 'multicollinearity_vif.png')
        plt.savefig(png_path, dpi=config.DPI)
        plt.close()
        logger.info(f"Multicollinearity (VIF) plot saved to: {png_path}")