import unittest
import pandas as pd
from analyzers.correlation_analyzer import CorrelationAnalyzer

class TestCorrelationAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = CorrelationAnalyzer()
        self.test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10],
            'C': [5, 4, 3, 2, 1]
        })

    def test_analyze(self):
        result = self.analyzer.analyze(self.test_data)
        expected = pd.DataFrame({
            'A': [1.0, 1.0, -1.0],
            'B': [1.0, 1.0, -1.0],
            'C': [-1.0, -1.0, 1.0]
        }, index=['A', 'B', 'C'])
        pd.testing.assert_frame_equal(result, expected)

    def test_analyze_with_non_numeric_column(self):
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': ['x', 'y', 'z']
        })
        result = self.analyzer.analyze(data)
        expected = pd.DataFrame({
            'A': [1.0, 1.0],
            'B': [1.0, 1.0]
        }, index=['A', 'B'])
        pd.testing.assert_frame_equal(result, expected)

    def test_analyze_single_column(self):
        data = pd.DataFrame({'A': [1, 2, 3]})
        result = self.analyzer.analyze(data)
        expected = pd.DataFrame({'A': [1.0]}, index=['A'])
        pd.testing.assert_frame_equal(result, expected)

    def test_analyze_empty_dataframe(self):
        data = pd.DataFrame()
        result = self.analyzer.analyze(data)
        self.assertTrue(result.empty)

if __name__ == '__main__':
    unittest.main()