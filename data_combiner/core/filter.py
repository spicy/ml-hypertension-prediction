import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..config import config
from ..logger import logger


class DataFilter:
    """Filters combined data based on relevant columns and age criteria."""

    def __init__(self, relevant_columns: List[str], min_age: int = 18):
        """Initialize filter with list of relevant columns and minimum age."""
        if not relevant_columns:
            raise ValueError("Relevant columns list cannot be empty")
        if not all(isinstance(col, str) for col in relevant_columns):
            raise ValueError("All column names must be strings")
        if min_age < 0:
            raise ValueError("Minimum age cannot be negative")

        self.relevant_columns = relevant_columns
        self.min_age = min_age

    def apply(
        self, df: pd.DataFrame, source_file: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Apply filtering to the DataFrame based on relevant columns and age criteria.
        Always places SEQN as the first column.
        """
        if df.empty:
            raise ValueError("Cannot filter empty DataFrame")

        # Get columns to keep
        columns_to_keep = set([config.SEQN_COLUMN])
        columns_to_keep.update(self.relevant_columns)

        # Validate columns exist
        missing_columns = columns_to_keep - set(df.columns)
        if missing_columns:
            logger.warning(f"Missing columns in data: {missing_columns}")
            columns_to_keep = columns_to_keep.intersection(set(df.columns))

        if not columns_to_keep:
            raise ValueError("No valid columns to keep after filtering")

        # Filter by age if RIDAGEYR column exists
        filtered_df = df.copy()
        if "RIDAGEYR" in filtered_df.columns:
            original_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df["RIDAGEYR"] >= self.min_age]
            filtered_count = len(filtered_df)
            logger.info(
                f"Age filter removed {original_count - filtered_count} entries younger than {self.min_age} years"
            )

        # Create ordered list of columns with SEQN first
        ordered_columns = [config.SEQN_COLUMN]
        ordered_columns.extend(
            sorted(col for col in columns_to_keep if col != config.SEQN_COLUMN)
        )

        return filtered_df[ordered_columns].copy()
