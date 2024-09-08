import os
import pandas as pd

def create_statistics_folder() -> str:
    print("Creating statistics folder...")
    statistics_folder = os.path.join(os.getcwd(), 'statistics')
    os.makedirs(statistics_folder, exist_ok=True)
    print(f"Statistics folder created at: {statistics_folder}")
    return statistics_folder

def load_data(file_path: str) -> pd.DataFrame:
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")

    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError(f"Error: The file '{file_path}' is empty.")
    except pd.errors.ParserError:
        raise ValueError(f"Error: Unable to parse '{file_path}'. Please check if it's a valid CSV file.")