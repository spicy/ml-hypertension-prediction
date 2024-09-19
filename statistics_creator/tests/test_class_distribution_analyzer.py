import unittest
import pandas as pd
from analyzers.class_distribution_analyzer import ClassDistributionAnalyzer

class TestClassDistributionAnalyzer(unittest.TestCase):
    def setUp(self):
        self.target_column = 'Class'
        self.analyzer = ClassDistributionAnalyzer(self.target_column)
        self.test_data = pd.DataFrame({
            'Class': ['A', 'B', 'A', 'C', 'B', 'A'],
            'Value': [1, 2, 3, 4, 5, 6]
        })

    def test_analyze(self):
        result = self.analyzer.analyze(self.test_data)
        expected = pd.Series({
            'A': 50.0,
            'B': 33.333333,
            'C': 16.666667
        })
        pd.testing.assert_series_equal(result, expected, check_less_precise=5)

    def test_analyze_single_class(self):
        data = pd.DataFrame({'Class': ['A', 'A', 'A'], 'Value': [1, 2, 3]})
        result = self.analyzer.analyze(data)
        expected = pd.Series({'A': 100.0})
        pd.testing.assert_series_equal(result, expected)

    def test_analyze_missing_target_column(self):
        data = pd.DataFrame({'Value': [1, 2, 3]})
        with self.assertRaises(ValueError):
            self.analyzer.analyze(data)

    def test_analyze_with_null_values(self):
        data = pd.DataFrame({'Class': ['A', 'B', None, 'A', 'B'], 'Value': [1, 2, 3, 4, 5]})
        result = self.analyzer.analyze(data)
        expected = pd.Series({'A': 50.0, 'B': 50.0})
        pd.testing.assert_series_equal(result, expected)

if __name__ == '__main__':
    unittest.main()