import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..config import config
from ..logger import logger
from .column_manager import ColumnManager


class DataFilter:
    """Filters combined data based on relevant columns and age criteria."""

    def __init__(self, relevant_columns: List[str], min_age: int = 18):
        """Initialize filter with list of relevant columns and minimum age."""
        if min_age < 0:
            raise ValueError("Minimum age cannot be negative")

        self.column_manager = ColumnManager(relevant_columns)
        self.min_age = min_age

    def apply(
        self, df: pd.DataFrame, source_file: Optional[Path] = None
    ) -> pd.DataFrame:
        """Apply filtering to the DataFrame."""
        if df.empty:
            raise ValueError("Cannot filter empty DataFrame")

        filtered_df = self.column_manager.ensure_relevant_columns_exist(df)

        if "RIDAGEYR" in filtered_df.columns:
            original_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df["RIDAGEYR"] >= self.min_age]
            filtered_count = len(filtered_df)
            logger.info(
                f"Age filter removed {original_count - filtered_count} entries younger than {self.min_age} years"
            )

        return filtered_df

    def validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that all required columns are present."""
        self.column_manager.validate_presence(df)
