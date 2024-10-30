import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..config import config
from ..logger import logger


class DataFilter:
    """Filters combined data based on relevant columns."""

    def __init__(self, relevant_columns: List[str]):
        """Initialize filter with list of relevant columns."""
        if not relevant_columns:
            raise ValueError("Relevant columns list cannot be empty")
        if not all(isinstance(col, str) for col in relevant_columns):
            raise ValueError("All column names must be strings")
        self.relevant_columns = relevant_columns

    def apply(
        self, df: pd.DataFrame, source_file: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Apply filtering to the DataFrame based on relevant columns.
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

        # Create ordered list of columns with SEQN first
        ordered_columns = [config.SEQN_COLUMN]
        ordered_columns.extend(
            sorted(col for col in columns_to_keep if col != config.SEQN_COLUMN)
        )

        return df[ordered_columns].copy()
