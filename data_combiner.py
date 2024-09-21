import pandas as pd
import json
from pathlib import Path
from typing import List, Optional

class DataCombiner:
    OUTPUT_DIR = Path('data') / '2017-2020' / 'processed'

    def __init__(self, input_files: List[str]):
        if not input_files:
            raise ValueError("Input files list cannot be empty.")
        self.input_files = input_files
        self.combined_df: Optional[pd.DataFrame] = None
        self.filtered_df: Optional[pd.DataFrame] = None

    def combine_data(self) -> None:
        dfs = []
        for file in self.input_files:
            try:
                df = pd.read_csv(file)
                if 'SEQN' not in df.columns:
                    raise ValueError(f"File {file} does not contain 'SEQN' column.")
                dfs.append(df)
            except Exception as e:
                print(f"Error reading file {file}: {str(e)}")

        if not dfs:
            raise ValueError("No valid dataframes to combine.")

        self.combined_df = dfs[0]
        for df in dfs[1:]:
            self.combined_df = pd.merge(self.combined_df, df, on='SEQN', how='outer')

    def filter_data(self, data_filter: 'DataFilter') -> None:
        if self.combined_df is None:
            raise ValueError("Data must be combined before filtering.")
        self.filtered_df = data_filter.apply(self.combined_df)

    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        if data.empty:
            raise ValueError("Cannot save empty DataFrame.")
        output_path = self.OUTPUT_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data.to_csv(output_path, index=False)
            print(f"Data has been saved to {output_path}")
        except Exception as e:
            print(f"Error saving data to {output_path}: {str(e)}")

    def save_combined_data(self, filename: str = 'UnfilteredCombinedData.csv') -> None:
        if self.combined_df is None:
            raise ValueError("No combined data to save.")
        self.save_data(self.combined_df, filename)

    def save_filtered_data(self, filename: str = 'FilteredCombinedData.csv') -> None:
        if self.filtered_df is None:
            raise ValueError("No filtered data to save.")
        self.save_data(self.filtered_df, filename)

class DataFilter:
    def __init__(self, relevant_columns: List[str]):
        if not relevant_columns:
            raise ValueError("Relevant columns list cannot be empty.")
        self.relevant_columns = relevant_columns

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_columns = set(self.relevant_columns) - set(df.columns)
        if missing_columns:
            print(f"Warning: The following columns are missing from the DataFrame: {missing_columns}")
        existing_columns = [col for col in self.relevant_columns if col in df.columns]
        return df[existing_columns]

def get_relevant_columns() -> List[str]:
    questions_dir = Path('questions')
    if not questions_dir.exists():
        raise FileNotFoundError(f"Questions directory not found: {questions_dir}")

    relevant_columns = ["SEQN"]

    for json_file in questions_dir.glob('*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                relevant_columns.extend(data.keys())
        except json.JSONDecodeError:
            print(f"Error decoding JSON file: {json_file}")
        except Exception as e:
            print(f"Error processing file {json_file}: {str(e)}")

    if len(relevant_columns) == 1:
        print("Warning: No additional columns found in JSON files.")

    return relevant_columns

def main():
    DATA_DIR = Path('data') / '2017-2020'
    RAW_DATA_DIR = DATA_DIR / 'raw'

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

    files_to_combine = list(RAW_DATA_DIR.glob('*.csv'))
    if not files_to_combine:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")

    relevant_columns = get_relevant_columns()

    try:
        combiner = DataCombiner(files_to_combine)
        combiner.combine_data()
        combiner.save_combined_data()

        data_filter = DataFilter(relevant_columns)
        combiner.filter_data(data_filter)
        combiner.save_filtered_data()
    except Exception as e:
        print(f"An error occurred during data processing: {str(e)}")

if __name__ == "__main__":
    main()