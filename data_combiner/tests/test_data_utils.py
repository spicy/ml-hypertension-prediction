import unittest
from typing import Any

import numpy as np
import pandas as pd

from ..utils.data_utils import convert_numeric_to_int64


class TestDataUtils(unittest.TestCase):
    """Test cases for data utility functions."""

    def test_convert_numeric_to_int64(self):
        """Test numeric column conversion to Int64."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, np.nan],
                "str_col": ["a", "b", "c"],
            }
        )

        converted_df = convert_numeric_to_int64(df)
        self.assertEqual(converted_df["int_col"].dtype, "Int64")
        self.assertNotEqual(converted_df["float_col"].dtype, "Int64")
        self.assertEqual(converted_df["str_col"].dtype, "object")
