import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Constants
SEQN_COLUMN = "SEQN"
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
QUESTIONS_DIR = Path("questions")
UNFILTERED_DATA_FILENAME = "UnfilteredCombinedData.csv"
FILTERED_DATA_FILENAME = "FilteredCombinedData.csv"


class DataCombiner:
    """A class for combining and filtering data from multiple CSV files."""

    OUTPUT_DIR = DATA_DIR / "processed"

    def __init__(self, input_files: List[Path]):
        """Initialize the DataCombiner with a list of input files."""
        if not input_files:
            raise ValueError("Input files list cannot be empty.")
        self.input_files = input_files
        self.combined_df: Optional[pd.DataFrame] = None
        self.filtered_df: Optional[pd.DataFrame] = None

    def combine_data(self) -> None:
        """Combine data from all input files into a single DataFrame."""
        dfs = []
        file_data = []  # Store tuples of (df, file_path)
        for file in self.input_files:
            try:
                # Read CSV and convert numeric columns to Int64 (nullable integer type)
                df = pd.read_csv(file)
                # Convert numeric columns to Int64 where possible
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        try:
                            df[col] = df[col].astype("Int64")
                        except (ValueError, TypeError):
                            # Keep as float if conversion fails
                            pass

                if SEQN_COLUMN not in df.columns:
                    raise ValueError(
                        f"File {file} does not contain '{SEQN_COLUMN}' column."
                    )
                file_data.append((df, file))
                dfs.append(df)
            except Exception as e:
                logging.error(f"Error reading file {file}: {str(e)}")

        if not dfs:
            raise ValueError("No valid dataframes to combine.")

        self.combined_df = pd.concat(dfs, join="outer", axis=1)
        self.combined_df = self.combined_df.loc[
            :, ~self.combined_df.columns.duplicated()
        ]
        # Store the source files for later reference
        self.source_files = [f for _, f in file_data]

    def filter_data(self, data_filter: "DataFilter") -> None:
        """Apply a filter to the combined data."""
        if self.combined_df is None:
            raise ValueError("Data must be combined before filtering.")
        # Pass the list of source files for context
        self.filtered_df = data_filter.apply(
            self.combined_df, self.source_files[0] if self.source_files else None
        )

    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        """Save a DataFrame to a CSV file."""
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
        """Save the combined data to a CSV file."""
        if self.combined_df is None:
            raise ValueError("No combined data to save.")
        self.save_data(self.combined_df, filename)

    def save_filtered_data(self, filename: str = FILTERED_DATA_FILENAME) -> None:
        """Save the filtered data to a CSV file."""
        if self.filtered_df is None:
            raise ValueError("No filtered data to save.")
        self.save_data(self.filtered_df, filename)


class DataFilter:
    """A class for filtering data based on relevant columns."""

    def __init__(self, relevant_columns: List[str]):
        """Initialize the DataFilter with a list of relevant columns."""
        if not relevant_columns:
            raise ValueError("Relevant columns list cannot be empty.")
        self.relevant_columns = relevant_columns

    def apply(
        self, df: pd.DataFrame, source_file: Optional[Path] = None
    ) -> pd.DataFrame:
        """Apply the filter to a DataFrame."""
        missing_columns = set(self.relevant_columns) - set(df.columns)
        if missing_columns:
            file_context = f" in {source_file}" if source_file else ""
            logging.warning(
                f"The following columns are missing from the DataFrame: {file_context}: {missing_columns}"
            )
        existing_columns = [col for col in self.relevant_columns if col in df.columns]
        return df[existing_columns]


def get_relevant_columns() -> List[str]:
    """Get a list of relevant columns from JSON files in the questions directory."""
    if not QUESTIONS_DIR.exists():
        raise FileNotFoundError(f"Questions directory not found: {QUESTIONS_DIR}")

    relevant_columns = set([SEQN_COLUMN])

    for json_file in QUESTIONS_DIR.glob("*.json"):
        try:
            with json_file.open("r") as f:
                data = json.load(f)
                relevant_columns.update(data.keys())
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON file: {json_file}")
        except Exception as e:
            logging.error(f"Error processing file {json_file}: {str(e)}")

    if len(relevant_columns) == 1:
        logging.warning("No additional columns found in JSON files.")

    return list(relevant_columns)


def get_data_directories() -> List[Path]:
    """Get all year-based data directories."""
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

    data_dirs = [
        d
        for d in RAW_DATA_DIR.iterdir()
        if d.is_dir() and d.name.replace("-", "").isdigit()
    ]
    if not data_dirs:
        raise FileNotFoundError(f"No year-based directories found in {RAW_DATA_DIR}")

    return data_dirs


def main():
    """Main function to execute the data combining and filtering process for all year directories."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    data_dirs = get_data_directories()
    relevant_columns = get_relevant_columns()

    for data_dir in data_dirs:
        if not data_dir.exists():
            logging.warning(f"Year directory not found: {data_dir}")
            continue

        files_to_combine = list(data_dir.glob("*.csv"))
        if not files_to_combine:
            logging.warning(f"No CSV files found in {data_dir}")
            continue

        try:
            # Create year-specific output filenames
            year_suffix = f"_{data_dir.name}"
            unfiltered_filename = UNFILTERED_DATA_FILENAME.replace(
                ".csv", f"{year_suffix}.csv"
            )
            filtered_filename = FILTERED_DATA_FILENAME.replace(
                ".csv", f"{year_suffix}.csv"
            )

            logging.info(f"Processing data for {data_dir.name}")
            combiner = DataCombiner(files_to_combine)
            combiner.combine_data()
            combiner.save_combined_data(unfiltered_filename)

            data_filter = DataFilter(relevant_columns)
            combiner.filter_data(data_filter)
            combiner.save_filtered_data(filtered_filename)

        except Exception as e:
            logging.exception(f"An error occurred processing {data_dir.name}: {str(e)}")


if __name__ == "__main__":
    main()
