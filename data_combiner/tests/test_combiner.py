import unittest
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from ..config import config
from ..core.combiner import DataCombiner


class TestDataCombiner(unittest.TestCase):
    """Test cases for DataCombiner class."""

    def setUp(self) -> None:
        """Set up test data."""
        self.test_files: List[Path] = [Path("ALQ_I.csv"), Path("BPQ_I.csv")]
        self.combiner: DataCombiner = DataCombiner(self.test_files)

    @patch("pandas.read_csv")
    def test_combine_data_success(self, mock_read_csv):
        # Realistic NHANES data structure
        alq_df = pd.DataFrame(
            {
                "SEQN": [83732, 83733],
                "ALQ101": [1, 1],
                "ALQ110": [np.nan, np.nan],
                "ALQ120Q": [1, 7],
                "ALQ120U": [2, 1],
            }
        )

        bpq_df = pd.DataFrame(
            {
                "SEQN": [83732, 83733, 83734],
                "BPQ020": [2, 2, 1],
                "BPQ030": [np.nan, np.nan, 1],
                "BPQ040A": [np.nan, np.nan, 1],
            }
        )

        mock_read_csv.side_effect = [alq_df, bpq_df]

        self.combiner.combine_data()

        self.assertIsNotNone(self.combiner.combined_df)
        self.assertEqual(
            len(self.combiner.combined_df), 3
        )  # Should include all unique SEQN
        self.assertTrue("ALQ101" in self.combiner.combined_df.columns)
        self.assertTrue("BPQ020" in self.combiner.combined_df.columns)

    @patch("pandas.read_csv")
    def test_combine_data_with_missing_values(self, mock_read_csv):
        alq_df = pd.DataFrame(
            {"SEQN": [83732, 83733], "ALQ101": [1, np.nan], "ALQ110": [np.nan, np.nan]}
        )

        bpq_df = pd.DataFrame(
            {"SEQN": [83732, 83734], "BPQ020": [2, 1], "BPQ030": [np.nan, 1]}
        )

        mock_read_csv.side_effect = [alq_df, bpq_df]
        self.combiner.combine_data()

        # Check that NaN values are preserved
        self.assertTrue(
            pd.isna(
                self.combiner.combined_df.loc[
                    self.combiner.combined_df["SEQN"] == 83733, "ALQ101"
                ]
            ).any()
        )
