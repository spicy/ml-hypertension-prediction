import unittest
import pandas as pd
import numpy as np
from statistics_creator.analyzers.multicollinearity_analyzer import MulticollinearityAnalyzer

class TestMulticollinearityAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = MulticollinearityAnalyzer()
        self.test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10],
            'C': [1, 2, 3, 4, 5]
        })

    def test_analyze(self):
        result = self.analyzer.analyze(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)  # Three features
        self.assertTrue(all(result['VIF'] >= 1))

    def test_analyze_perfect_collinearity(self):
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [2, 4, 6]
        })
        result = self.analyzer.analyze(data)
        self.assertTrue(all(result['VIF'] > 1000))  # Very high VIF for perfect collinearity

    def test_analyze_no_collinearity(self):
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        result = self.analyzer.analyze(data)
        self.assertTrue(all(result['VIF'] < 5))  # Low VIF for no collinearity

if __name__ == '__main__':
    unittest.main()