import unittest
import pandas as pd
import os
from statistics_creator.visualizers.summary_statistics_visualizer import SummaryStatisticsVisualizer

class TestSummaryStatisticsVisualizer(unittest.TestCase):
    def setUp(self):
        self.visualizer = SummaryStatisticsVisualizer()
        self.test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        }).describe()
        self.output_path = 'test_output'
        os.makedirs(self.output_path, exist_ok=True)

    def test_visualize(self):
        self.visualizer.visualize(self.test_data, self.output_path)
        self.assertTrue(os.path.exists(os.path.join(self.output_path, 'summary_statistics_heatmap.png')))

    def tearDown(self):
        for file in os.listdir(self.output_path):
            os.remove(os.path.join(self.output_path, file))
        os.rmdir(self.output_path)

if __name__ == '__main__':
    unittest.main()