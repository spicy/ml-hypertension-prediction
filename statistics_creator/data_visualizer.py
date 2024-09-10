import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from constants import *
from logger import logger, log_execution_time

@log_execution_time
def plot_missing_data(missing_percentage_sorted: pd.Series, statistics_folder: str) -> None:
    logger.info("Plotting missing data...")
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    ax = sns.barplot(x=missing_percentage_sorted.index, y=missing_percentage_sorted.values)

    plt.title('Percentage of Missing Data by Column', fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    plt.xlabel('Columns', fontsize=XLABEL_FONTSIZE, labelpad=LABEL_PAD)
    plt.ylabel('Percentage of Missing Data', fontsize=YLABEL_FONTSIZE, labelpad=LABEL_PAD)
    plt.xticks(rotation=90, fontsize=XTICK_FONTSIZE)
    plt.yticks(fontsize=YTICK_FONTSIZE)

    for i, v in enumerate(missing_percentage_sorted.values):
        ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=TEXT_FONTSIZE)

    plt.ylim(0, max(missing_percentage_sorted) * YLIM_MULTIPLIER)
    plt.tight_layout(pad=TIGHT_LAYOUT_PAD)

    png_path = os.path.join(statistics_folder, 'missing_data_percentage.png')
    plt.savefig(png_path, dpi=DPI)
    plt.close()
    logger.info(f"Missing data plot saved to: {png_path}")

@log_execution_time
def plot_correlation_matrix(correlation_matrix: pd.DataFrame, statistics_folder: str):
    logger.info("Plotting correlation matrix...")
    plt.figure(figsize=(CORRELATION_FIGURE_WIDTH, CORRELATION_FIGURE_HEIGHT))

    sns.heatmap(correlation_matrix,
                cmap='RdBu',
                vmin=-1,
                vmax=1,
                center=0,
                annot=True,
                fmt='.2f',
                square=True,
                cbar_kws={"shrink": .8},
                annot_kws={"size": CORRELATION_ANNOT_FONTSIZE})

    plt.title('Correlation Matrix Heatmap', fontsize=CORRELATION_TITLE_FONTSIZE, pad=TITLE_PAD)
    plt.xticks(rotation=90, fontsize=CORRELATION_XTICK_FONTSIZE)
    plt.yticks(rotation=0, fontsize=CORRELATION_YTICK_FONTSIZE)

    plt.tight_layout(pad=CORRELATION_TIGHT_LAYOUT_PAD)

    png_path = os.path.join(statistics_folder, 'correlation_matrix_heatmap.png')
    plt.savefig(png_path, dpi=DPI)
    plt.close()
    logger.info(f"Correlation matrix plot saved to: {png_path}")