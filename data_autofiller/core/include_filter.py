from typing import Dict, List

import pandas as pd

from ..logger import logger


class IncludeFilter:
    """Filters data to only include columns marked with include: 1 in questions data."""

    def __init__(self, questions_data: Dict):
        """Initialize filter with questions data."""
        self.included_columns = self._get_included_columns(questions_data)
        logger.debug(f"Columns to include: {self.included_columns}")

    def _get_included_columns(self, questions_data: Dict) -> List[str]:
        """Extract column names that have include: 1."""
        return [
            col
            for col, data in questions_data.items()
            if data.get("include") == "1" or data.get("include") == 1
        ]

    def _get_ordered_columns(self, available_columns: set) -> List[str]:
        """Get ordered list of columns with SEQN first."""
        seqn_column = "SEQN"
        ordered_columns = [seqn_column] if seqn_column in available_columns else []
        other_columns = sorted(col for col in available_columns if col != seqn_column)
        return ordered_columns + other_columns

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply include filter to the DataFrame."""
        if df.empty:
            logger.warning("Cannot filter empty DataFrame")
            return df

        # Always keep the SEQN column if it exists
        columns_to_keep = set(self.included_columns)
        if "SEQN" in df.columns:
            columns_to_keep.add("SEQN")

        # Get intersection of columns to keep and available columns
        available_columns = columns_to_keep.intersection(df.columns)

        if not available_columns:
            logger.warning("No included columns found in DataFrame")
            return df

        # Get ordered columns and filter DataFrame
        ordered_columns = self._get_ordered_columns(available_columns)
        filtered_df = df[ordered_columns]

        logger.info(
            f"Include filter removed {len(df.columns) - len(filtered_df.columns)} columns. "
            f"Remaining columns: {ordered_columns}"
        )

        return filtered_df
