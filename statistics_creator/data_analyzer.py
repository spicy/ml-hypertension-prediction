import pandas as pd
import numpy as np
from logger import logger, log_execution_time

@log_execution_time
def calculate_class_distribution(df: pd.DataFrame, target_column: str) -> pd.Series:
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in the dataset.")
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    target_values = df[target_column].dropna()  # Drop missing values
    value_counts = target_values.value_counts()
    total = len(target_values)

    logger.info(f"Class distribution calculated for column '{target_column}'")
    return (value_counts / total) * 100

@log_execution_time
def analyze_missing_data(df: pd.DataFrame) -> pd.Series:
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_percentage_sorted = missing_percentage.sort_values(ascending=False)
    logger.info("Missing data analysis completed.")
    return missing_percentage_sorted

@log_execution_time
def calculate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    logger.info("Correlation matrix calculation completed.")
    return correlation_matrix

@log_execution_time
def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    summary_statistics = df.describe()
    logger.info("Summary statistics generation completed.")
    return summary_statistics