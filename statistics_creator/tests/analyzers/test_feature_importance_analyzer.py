import unittest
import pandas as pd
import numpy as np
from statistics_creator.analyzers.feature_importance_analyzer import FeatureImportanceAnalyzer

class TestFeatureImportanceAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = FeatureImportanceAnalyzer('target')
        self.test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target': [0, 1, 0, 1, 1]
        })

    def test_analyze(self):
        result = self.analyzer.analyze(self.test_data)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 2)  # Two features
        self.assertTrue(all(0 <= importance <= 1 for importance in result))

    def test_analyze_no_features(self):
        data = pd.DataFrame({'target': [0, 1, 0, 1, 1]})
        with self.assertRaises(ValueError):
            self.analyzer.analyze(data)

    def test_analyze_missing_target(self):
        data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        with self.assertRaises(ValueError):
            self.analyzer.analyze(data)

if __name__ == '__main__':
    unittest.main()