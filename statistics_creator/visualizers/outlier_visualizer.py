import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .base_visualizer import BaseVisualizer
from config import outlier_config as config
from logger import logger, log_execution_time

class OutlierVisualizer(BaseVisualizer):
    @log_execution_time
    def visualize(self, outlier_data: dict, output_path: str) -> None:
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
            plt.axhline(y=data['lower_bound'], color='r', linestyle='--', label='Lower Bound')
            plt.axhline(y=data['upper_bound'], color='g', linestyle='--', label='Upper Bound')
            plt.legend(fontsize=config.LEGEND_FONT_SIZE)

            plt.tight_layout()

            # Save the plot
            png_path = os.path.join(output_path, f'outliers_{column}.png')
            plt.savefig(png_path, dpi=config.DPI)
            plt.close()

        logger.info(f"Outlier plots saved to: {output_path}")