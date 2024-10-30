import unittest
from pathlib import Path

import pandas as pd

from ..config import config
from ..core.filter import DataFilter


class TestDataFilter(unittest.TestCase):
    """Test cases for DataFilter class."""

    def setUp(self):
        """Set up test data."""
        self.test_df = pd.DataFrame(
            {
                "SEQN": [1, 2, 3],
                "COL1": ["a", "b", "c"],
                "COL2": [1, 2, 3],
                "COL3": [4, 5, 6],
            }
        )
        self.relevant_columns = ["COL1", "COL2"]
        self.filter = DataFilter(self.relevant_columns)

    def test_filter_columns(self):
        """Test basic column filtering."""
        filtered_df = self.filter.apply(self.test_df)
        expected_columns = {config.SEQN_COLUMN, "COL1", "COL2"}
        self.assertEqual(set(filtered_df.columns), expected_columns)

    def test_filter_missing_columns(self):
        """Test filtering with missing columns."""
        self.filter.relevant_columns.append("MISSING_COL")
        filtered_df = self.filter.apply(self.test_df)
        self.assertIn(config.SEQN_COLUMN, filtered_df.columns)
        self.assertIn("COL1", filtered_df.columns)
        self.assertIn("COL2", filtered_df.columns)
        self.assertNotIn("MISSING_COL", filtered_df.columns)

    def test_empty_dataframe(self):
        """Test filtering empty DataFrame."""
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.filter.apply(empty_df)

    def test_no_valid_columns(self):
        """Test filtering with no valid columns."""
        self.filter.relevant_columns = ["MISSING1", "MISSING2"]
        with self.assertRaises(ValueError):
            self.filter.apply(self.test_df)

    def test_seqn_first_column(self):
        """Test that SEQN is always the first column."""
        filtered_df = self.filter.apply(self.test_df)
        self.assertEqual(filtered_df.columns[0], config.SEQN_COLUMN)
        self.assertEqual(
            list(filtered_df.columns), [config.SEQN_COLUMN, "COL1", "COL2"]
        )
