from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..config import config
from ..logger import logger
from ..utils.data_utils import convert_numeric_to_int64
from ..utils.file_utils import read_and_validate_file
from .filter import DataFilter
from .validator import DataValidator


class DataCombiner:
    """A class for combining data from multiple CSV files."""

    def __init__(self, input_files: List[Path]):
        """Initialize the DataCombiner with a list of input files."""
        if not input_files:
            raise ValueError("Input files list cannot be empty.")
        self.input_files = input_files
        self.combined_df: Optional[pd.DataFrame] = None
        self.filtered_df: Optional[pd.DataFrame] = None
        self.source_files: List[Path] = []

    def combine_data(self) -> None:
        """Combine data from all input files using SEQN-based merges."""
        if not self.input_files:
            raise ValueError("No input files to process")

        try:
            base_df = read_and_validate_file(self.input_files[0])
            self.source_files = [self.input_files[0]]

            for file in self.input_files[1:]:
                try:
                    df = read_and_validate_file(file)
                    base_df = self._merge_dataframes(base_df, df)
                    self.source_files.append(file)
                except FileNotFoundError as e:
                    logger.error(f"File not found: {file}")
                    continue
                except ValueError as e:
                    logger.error(f"Invalid file format in {file}: {str(e)}")
                    continue
                except pd.errors.EmptyDataError:
                    logger.error(f"Empty file: {file}")
                    continue
                except pd.errors.ParserError as e:
                    logger.error(f"Error parsing {file}: {str(e)}")
                    continue

            if base_df.empty:
                raise ValueError("No data was successfully combined")

            self.combined_df = base_df
            logger.info(
                f"Combined data contains {len(base_df[config.SEQN_COLUMN].unique())} unique SEQN values"
            )

        except Exception as e:
            logger.error(f"Unexpected error in combine_data: {str(e)}")
            raise

    def _merge_dataframes(
        self, base_df: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge two DataFrames based on SEQN."""
        merged_df = base_df.merge(
            df, on=config.SEQN_COLUMN, how="outer", suffixes=("", "_duplicate")
        )

        duplicate_cols = [
            col for col in merged_df.columns if col.endswith("_duplicate")
        ]
        merged_df = merged_df.drop(columns=duplicate_cols)

        if merged_df[config.SEQN_COLUMN].isna().any():
            logger.warning("Found empty SEQN values after merge")

        return merged_df

    def save_combined_data(
        self, filename: str = config.UNFILTERED_DATA_FILENAME
    ) -> None:
        """Save the combined data to a CSV file."""
        if self.combined_df is None:
            raise ValueError("No combined data to save.")
        output_path = config.PROCESSED_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.combined_df.to_csv(output_path, index=False)
        logger.info(f"Combined data saved to {output_path}")

    def validate_combined_data(self, unfiltered_path: Path) -> None:
        """Validate the combined data against source files."""
        if self.combined_df is None:
            raise ValueError("No combined data to validate")

        validator = DataValidator(self.combined_df, self.source_files)
        validator.validate()
        logger.info("Combined data validation successful")

    def filter_data(self, data_filter: DataFilter) -> None:
        """Apply filtering to the combined data."""
        if self.combined_df is None:
            raise ValueError("No combined data to filter")

        self.filtered_df = data_filter.apply(self.combined_df)
        logger.info(
            f"Filtered data from {len(self.combined_df.columns)} to {len(self.filtered_df.columns)} columns"
        )

    def save_filtered_data(self, filename: str = config.FILTERED_DATA_FILENAME) -> None:
        """Save the filtered data to a CSV file."""
        if self.filtered_df is None:
            raise ValueError("No filtered data to save")

        output_path = config.PROCESSED_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.filtered_df.to_csv(output_path, index=False)
        logger.info(f"Filtered data saved to {output_path}")
