import unittest
import os
from statistics_creator.visualizers.outlier_visualizer import OutlierVisualizer

class TestOutlierVisualizer(unittest.TestCase):
    def setUp(self):
        self.visualizer = OutlierVisualizer()
        self.test_data = {
            'A': {'outliers': [1, 10], 'lower_bound': 2, 'upper_bound': 8},
            'B': {'outliers': [20, 30], 'lower_bound': 5, 'upper_bound': 15}
        }
        self.output_path = 'test_output'
        os.makedirs(self.output_path, exist_ok=True)

    def test_visualize(self):
        self.visualizer.visualize(self.test_data, self.output_path)
        self.assertTrue(os.path.exists(os.path.join(self.output_path, 'outliers_A.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_path, 'outliers_B.png')))

    def tearDown(self):
        for file in os.listdir(self.output_path):
            os.remove(os.path.join(self.output_path, file))
        os.rmdir(self.output_path)

if __name__ == '__main__':
    unittest.main()