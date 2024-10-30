import unittest
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from ..config import config
from ..core.validator import DataValidator, ValidationError


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator class."""

    def setUp(self) -> None:
        """Set up test data."""
        self.test_df: pd.DataFrame = pd.DataFrame(
            {
                "SEQN": [83732, 83733, 83734],
                "ALQ101": [1, 2, np.nan],
                "BPQ020": [2, 1, 1],
            }
        )
        self.source_files: List[Path] = [Path("ALQ_I.csv"), Path("BPQ_I.csv")]
        self.validator: DataValidator = DataValidator(self.test_df, self.source_files)

    @patch("pandas.read_csv")
    def test_validate_matching_data(self, mock_read_csv):
        source_df = pd.DataFrame(
            {"SEQN": [83732, 83733], "ALQ101": [1, 2], "ALQ110": [np.nan, 3]}
        )
        mock_read_csv.return_value = source_df

        self.validator.validate()
        self.assertEqual(len(self.validator.validation_errors), 0)

    @patch("pandas.read_csv")
    def test_validate_mismatched_data(self, mock_read_csv):
        source_df = pd.DataFrame(
            {
                "SEQN": [83732, 83733],
                "ALQ101": [1, 3],  # Value mismatch for SEQN 83733
                "ALQ110": [np.nan, 3],
            }
        )
        mock_read_csv.return_value = source_df

        with self.assertRaises(ValueError) as context:
            self.validator.validate()
        self.assertTrue("validation errors" in str(context.exception))

    @patch("pandas.read_csv")
    def test_validate_missing_seqn(self, mock_read_csv):
        """Test validation when SEQN is missing from combined data."""
        source_df = pd.DataFrame(
            {"SEQN": [83732, 83735], "ALQ101": [1, 2]}  # 83735 not in combined data
        )
        mock_read_csv.return_value = source_df

        with self.assertRaises(ValueError):
            self.validator.validate()

        error = next(
            e
            for e in self.validator.validation_errors
            if e.error_type == "missing_seqn"
        )
        self.assertEqual(error.seqn, 83735)

    @patch("pandas.read_csv")
    def test_validate_no_common_columns(self, mock_read_csv):
        """Test validation when no common columns exist."""
        source_df = pd.DataFrame({"SEQN": [83732], "DIFFERENT_COL": [1]})
        mock_read_csv.return_value = source_df

        with self.assertRaises(ValueError):
            self.validator.validate()

        error = next(
            e
            for e in self.validator.validation_errors
            if e.error_type == "no_common_columns"
        )
        self.assertIsNotNone(error)

    @patch("pandas.read_csv")
    def test_file_read_error(self, mock_read_csv):
        """Test handling of file read errors."""
        mock_read_csv.side_effect = pd.errors.EmptyDataError()

        with self.assertRaises(Exception):
            self.validator.validate()
