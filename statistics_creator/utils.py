import os
import pandas as pd
from logger import logger, log_execution_time

@log_execution_time
def save_class_distribution(class_distribution: pd.Series, statistics_folder: str):
    logger.info("Saving class distribution...")
    class_distribution_df = pd.DataFrame({
        'Class': class_distribution.index,
        'Percentage': class_distribution.values
    })

    csv_path = os.path.join(statistics_folder, 'class_distribution.csv')
    class_distribution_df.to_csv(csv_path, index=False)
    logger.info(f"Class distribution saved to: {csv_path}")

@log_execution_time
def save_missing_data_summary(missing_percentage: pd.Series, statistics_folder: str):
    logger.info("Saving missing data summary...")
    missing_summary = pd.DataFrame({
        'Column': missing_percentage.index,
        'Percentage Missing': missing_percentage.values
    })
    missing_summary = missing_summary.sort_values('Percentage Missing', ascending=False)
    csv_path = os.path.join(statistics_folder, 'missing_data_summary.csv')
    missing_summary.to_csv(csv_path, index=False)
    logger.info(f"Missing data summary saved to: {csv_path}")

@log_execution_time
def save_summary_statistics(summary_statistics: pd.DataFrame, statistics_folder: str):
    logger.info("Saving summary statistics...")
    csv_path = os.path.join(statistics_folder, 'summary_statistics.csv')
    summary_statistics.to_csv(csv_path)
    logger.info(f"Summary statistics saved to: {csv_path}")