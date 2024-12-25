from typing import List, Set

import pandas as pd

from ..config import config
from ..logger import logger


class ColumnManager:
    """Manages column operations for DataFrames."""

    def __init__(self, relevant_columns: List[str]):
        if not relevant_columns:
            raise ValueError("Relevant columns list cannot be empty")
        self.relevant_columns = set(relevant_columns)
        self._validate_columns()

    def _validate_columns(self) -> None:
        """Validate column names."""
        if not all(isinstance(col, str) for col in self.relevant_columns):
            raise ValueError("All column names must be strings")
        if config.SEQN_COLUMN not in self.relevant_columns:
            self.relevant_columns.add(config.SEQN_COLUMN)

    def ensure_relevant_columns_exist(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all relevant columns exist in DataFrame."""
        df_copy = df.copy()
        missing_columns = self.relevant_columns - set(df_copy.columns)

        if missing_columns:
            logger.warning(f"Adding missing columns with NaN values: {missing_columns}")
            for col in missing_columns:
                df_copy[col] = pd.NA

        return df_copy[self.get_ordered_columns()]

    def get_ordered_columns(self) -> List[str]:
        """Get ordered list of columns with SEQN first."""
        return [config.SEQN_COLUMN] + sorted(
            col for col in self.relevant_columns if col != config.SEQN_COLUMN
        )

    def validate_presence(self, df: pd.DataFrame) -> None:
        """Validate presence of columns in DataFrame."""
        missing_columns = self.relevant_columns - set(df.columns)
        if missing_columns:
            logger.warning(
                f"The following columns from questions are not present in the data: {sorted(missing_columns)}"
            )
