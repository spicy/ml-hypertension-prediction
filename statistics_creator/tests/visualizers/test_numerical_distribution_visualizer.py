import unittest
import os
from statistics_creator.visualizers.numerical_distribution_visualizer import NumericalDistributionVisualizer

class TestNumericalDistributionVisualizer(unittest.TestCase):
    def setUp(self):
        self.visualizer = NumericalDistributionVisualizer()
        self.test_data = {
            'A': {'mean': 3, 'median': 3, 'std': 1.4, 'skewness': 0, 'kurtosis': -1.2, 'min': 1, 'max': 5, 'q1': 2, 'q3': 4},
            'B': {'mean': 6, 'median': 6, 'std': 2.8, 'skewness': 0, 'kurtosis': -1.2, 'min': 2, 'max': 10, 'q1': 4, 'q3': 8}
        }
        self.output_path = 'test_output'
        os.makedirs(self.output_path, exist_ok=True)

    def test_visualize(self):
        self.visualizer.visualize(self.test_data, self.output_path)
        self.assertTrue(os.path.exists(os.path.join(self.output_path, 'numerical_distribution_A.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_path, 'numerical_distribution_B.png')))

    def tearDown(self):
        for file in os.listdir(self.output_path):
            os.remove(os.path.join(self.output_path, file))
        os.rmdir(self.output_path)

if __name__ == '__main__':
    unittest.main()