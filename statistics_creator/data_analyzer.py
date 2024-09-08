import pandas as pd

def analyze_missing_data(df: pd.DataFrame) -> pd.Series:
    print("Analyzing missing data...")
    missing_percentage = df.isnull().mean() * 100
    missing_percentage_sorted = missing_percentage.sort_values(ascending=False)
    print("Missing data analysis completed.")
    return missing_percentage_sorted

def calculate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    print("Calculating correlation matrix...")
    correlation_matrix = df.corr()
    print("Correlation matrix calculation completed.")
    return correlation_matrix

def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    print("Generating summary statistics...")
    summary_statistics = df.describe()
    print("Summary statistics generation completed.")
    return summary_statistics