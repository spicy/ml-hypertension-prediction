import unittest
import pandas as pd
import numpy as np
from statistics_creator.analyzers.outlier_analyzer import OutlierAnalyzer

class TestOutlierAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = OutlierAnalyzer()
        self.test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 100],
            'B': [2, 4, 6, 8, 10, 12],
            'C': ['x', 'y', 'z', 'x', 'y', 'z']
        })

    def test_analyze(self):
        result = self.analyzer.analyze(self.test_data)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)  # Two numerical columns
        for col_data in result.values():
            self.assertIn('lower_bound', col_data)
            self.assertIn('upper_bound', col_data)
            self.assertIn('outliers', col_data)

    def test_analyze_no_outliers(self):
        data = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        result = self.analyzer.analyze(data)
        self.assertEqual(len(result['A']['outliers']), 0)

    def test_analyze_with_nan_values(self):
        data = pd.DataFrame({'A': [1, 2, np.nan, 4, 5, 100]})
        result = self.analyzer.analyze(data)
        self.assertIsInstance(result['A'], dict)
        self.assertTrue(100 in result['A']['outliers'])

if __name__ == '__main__':
    unittest.main()