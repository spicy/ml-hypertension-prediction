import unittest
import pandas as pd
from analyzers.missing_data_analyzer import MissingDataAnalyzer

class TestMissingDataAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = MissingDataAnalyzer()
        self.test_data = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [5, None, None, 8],
            'C': [9, 10, 11, 12]
        })

    def test_analyze(self):
        result = self.analyzer.analyze(self.test_data)
        expected = pd.Series({
            'B': 50.0,
            'A': 25.0,
            'C': 0.0
        })
        pd.testing.assert_series_equal(result, expected)

    def test_analyze_no_missing_data(self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = self.analyzer.analyze(data)
        expected = pd.Series({'A': 0.0, 'B': 0.0})
        pd.testing.assert_series_equal(result, expected)

    def test_analyze_all_missing_data(self):
        data = pd.DataFrame({'A': [None, None], 'B': [None, None]})
        result = self.analyzer.analyze(data)
        expected = pd.Series({'A': 100.0, 'B': 100.0})
        pd.testing.assert_series_equal(result, expected)

if __name__ == '__main__':
    unittest.main()