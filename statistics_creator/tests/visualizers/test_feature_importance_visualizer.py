import unittest
import pandas as pd
import os
from statistics_creator.visualizers.feature_importance_visualizer import FeatureImportanceVisualizer

class TestImportanceVisualizer(unittest.TestCase):
    def setUp(self):
        self.visualizer = FeatureImportanceVisualizer()
        self.test_data = pd.Series({'feature1': 0.5, 'feature2': 0.3, 'feature3': 0.2})
        self.output_path = 'test_output'
        os.makedirs(self.output_path, exist_ok=True)

    def test_visualize(self):
        self.visualizer.visualize(self.test_data, self.output_path)
        self.assertTrue(os.path.exists(os.path.join(self.output_path, 'feature_importance.png')))

    def tearDown(self):
        for file in os.listdir(self.output_path):
            os.remove(os.path.join(self.output_path, file))
        os.rmdir(self.output_path)

if __name__ == '__main__':
    unittest.main()