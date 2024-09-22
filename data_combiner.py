import pandas as pd
import json
from pathlib import Path
from typing import List, Optional
import logging

# Constants
SEQN_COLUMN = 'SEQN'
DATA_DIR = Path('data') / '2017-2020'
RAW_DATA_DIR = DATA_DIR / 'raw'
QUESTIONS_DIR = Path('questions')
UNFILTERED_DATA_FILENAME = 'UnfilteredCombinedData.csv'
FILTERED_DATA_FILENAME = 'FilteredCombinedData.csv'

class DataCombiner:
    """
    A class for combining and filtering data from multiple CSV files.

    Attributes:
        OUTPUT_DIR (Path): The directory where processed data will be saved.
        input_files (List[Path]): List of input CSV file paths.
        combined_df (Optional[pd.DataFrame]): The combined DataFrame.
        filtered_df (Optional[pd.DataFrame]): The filtered DataFrame.
    """

    OUTPUT_DIR = DATA_DIR / 'processed'

    def __init__(self, input_files: List[Path]):
        """
        Initialize the DataCombiner with a list of input files.

        Args:
            input_files (List[Path]): List of input CSV file paths.

        Raises:
            ValueError: If the input files list is empty.
        """
        if not input_files:
            raise ValueError("Input files list cannot be empty.")
        self.input_files = input_files
        self.combined_df: Optional[pd.DataFrame] = None
        self.filtered_df: Optional[pd.DataFrame] = None

    def combine_data(self) -> None:
        """
        Combine data from all input files into a single DataFrame.

        Raises:
            ValueError: If no valid dataframes are available to combine.
        """
        dfs = []
        for file in self.input_files:
            try:
                df = pd.read_csv(file)
                if SEQN_COLUMN not in df.columns:
                    raise ValueError(f"File {file} does not contain '{SEQN_COLUMN}' column.")
                dfs.append(df)
            except Exception as e:
                logging.error(f"Error reading file {file}: {str(e)}")

        if not dfs:
            raise ValueError("No valid dataframes to combine.")

        self.combined_df = pd.concat(dfs, join='outer', axis=1)
        self.combined_df = self.combined_df.loc[:, ~self.combined_df.columns.duplicated()]

    def filter_data(self, data_filter: 'DataFilter') -> None:
        """
        Apply a filter to the combined data.

        Args:
            data_filter (DataFilter): The filter to apply to the data.

        Raises:
            ValueError: If data has not been combined before filtering.
        """
        if self.combined_df is None:
            raise ValueError("Data must be combined before filtering.")
        self.filtered_df = data_filter.apply(self.combined_df)

    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save a DataFrame to a CSV file.

        Args:
            data (pd.DataFrame): The DataFrame to save.
            filename (str): The name of the file to save the data to.

        Raises:
            ValueError: If the DataFrame is empty.
        """
        if data.empty:
            raise ValueError("Cannot save empty DataFrame.")
        output_path = self.OUTPUT_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data.to_csv(output_path, index=False)
            logging.info(f"Data has been saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving data to {output_path}: {str(e)}")

    def save_combined_data(self, filename: str = UNFILTERED_DATA_FILENAME) -> None:
        """
        Save the combined data to a CSV file.

        Args:
            filename (str, optional): The name of the file to save the combined data to.
                Defaults to UNFILTERED_DATA_FILENAME.

        Raises:
            ValueError: If no combined data is available to save.
        """
        if self.combined_df is None:
            raise ValueError("No combined data to save.")
        self.save_data(self.combined_df, filename)

    def save_filtered_data(self, filename: str = FILTERED_DATA_FILENAME) -> None:
        """
        Save the filtered data to a CSV file.

        Args:
            filename (str, optional): The name of the file to save the filtered data to.
                Defaults to FILTERED_DATA_FILENAME.

        Raises:
            ValueError: If no filtered data is available to save.
        """
        if self.filtered_df is None:
            raise ValueError("No filtered data to save.")
        self.save_data(self.filtered_df, filename)

class DataFilter:
    """
    A class for filtering data based on relevant columns.

    Attributes:
        relevant_columns (List[str]): List of column names to keep in the filtered data.
    """

    def __init__(self, relevant_columns: List[str]):
        """
        Initialize the DataFilter with a list of relevant columns.

        Args:
            relevant_columns (List[str]): List of column names to keep in the filtered data.

        Raises:
            ValueError: If the relevant columns list is empty.
        """
        if not relevant_columns:
            raise ValueError("Relevant columns list cannot be empty.")
        self.relevant_columns = relevant_columns

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the filter to a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to filter.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        missing_columns = set(self.relevant_columns) - set(df.columns)
        if missing_columns:
            print(f"Warning: The following columns are missing from the DataFrame: {missing_columns}")
        existing_columns = [col for col in self.relevant_columns if col in df.columns]
        return df[existing_columns]

def get_relevant_columns() -> List[str]:
    """
    Get a list of relevant columns from JSON files in the questions directory.

    Returns:
        List[str]: A list of relevant column names.

    Raises:
        FileNotFoundError: If the questions directory is not found.
    """
    if not QUESTIONS_DIR.exists():
        raise FileNotFoundError(f"Questions directory not found: {QUESTIONS_DIR}")

    relevant_columns = set([SEQN_COLUMN])

    for json_file in QUESTIONS_DIR.glob('*.json'):
        try:
            with json_file.open('r') as f:
                data = json.load(f)
                relevant_columns.update(data.keys())
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON file: {json_file}")
        except Exception as e:
            logging.error(f"Error processing file {json_file}: {str(e)}")

    if len(relevant_columns) == 1:
        logging.warning("No additional columns found in JSON files.")

    return list(relevant_columns)

def main():
    """
    Main function to execute the data combining and filtering process.

    Raises:
        FileNotFoundError: If the raw data directory or CSV files are not found.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        logging.exception(f"An error occurred during data processing: {str(e)}")

if __name__ == "__main__":
    main()