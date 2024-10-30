import unittest

import numpy as np
import pandas as pd

from statistics_creator.analyzers.numerical_distribution_analyzer import (
    NumericalDistributionAnalyzer,
)


class TestNumericalDistributionAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = NumericalDistributionAnalyzer()
        self.test_data = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [2, 4, 6, 8, 10],
                "C": ["x", "y", "z", "x", "y"],
            }
        )

    def test_analyze(self):
        result = self.analyzer.analyze(self.test_data)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)  # Two numerical columns
        for col_stats in result.values():
            self.assertIsInstance(col_stats, dict)
            self.assertTrue(
                all(
                    key in col_stats
                    for key in [
                        "mean",
                        "median",
                        "std",
                        "skewness",
                        "kurtosis",
                        "min",
                        "max",
                        "q1",
                        "q3",
                    ]
                )
            )

    def test_analyze_no_numerical_columns(self):
        data = pd.DataFrame({"A": ["x", "y", "z"], "B": ["a", "b", "c"]})
        result = self.analyzer.analyze(data)
        self.assertEqual(len(result), 0)

    def test_analyze_with_nan_values(self):
        data = pd.DataFrame({"A": [1, 2, np.nan, 4, 5]})
        result = self.analyzer.analyze(data)
        self.assertIsInstance(result["A"], dict)
        self.assertTrue(all(not np.isnan(value) for value in result["A"].values()))


if __name__ == "__main__":
    unittest.main()
