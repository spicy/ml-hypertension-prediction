import unittest
import pandas as pd
import os
from statistics_creator.visualizers.missing_data_visualizer import MissingDataVisualizer

class TestMissingDataVisualizer(unittest.TestCase):
    def setUp(self):
        self.visualizer = MissingDataVisualizer()
        self.test_data = pd.Series({'A': 10, 'B': 20, 'C': 30})
        self.output_path = 'test_output'
        os.makedirs(self.output_path, exist_ok=True)

    def test_visualize(self):
        self.visualizer.visualize(self.test_data, self.output_path)
        self.assertTrue(os.path.exists(os.path.join(self.output_path, 'missing_data.png')))

    def tearDown(self):
        for file in os.listdir(self.output_path):
            os.remove(os.path.join(self.output_path, file))
        os.rmdir(self.output_path)

if __name__ == '__main__':
    unittest.main()