import unittest
import pandas as pd
import os
from statistics_creator.visualizers.class_distribution_visualizer import ClassDistributionVisualizer

class TestClassDistributionVisualizer(unittest.TestCase):
    def setUp(self):
        self.visualizer = ClassDistributionVisualizer()
        self.test_data = pd.Series({'Class A': 40, 'Class B': 30, 'Class C': 30})
        self.output_path = 'test_output'
        os.makedirs(self.output_path, exist_ok=True)

    def test_visualize(self):
        self.visualizer.visualize(self.test_data, self.output_path)
        self.assertTrue(os.path.exists(os.path.join(self.output_path, 'class_distribution.png')))

    def tearDown(self):
        for file in os.listdir(self.output_path):
            os.remove(os.path.join(self.output_path, file))
        os.rmdir(self.output_path)

if __name__ == '__main__':
    unittest.main()