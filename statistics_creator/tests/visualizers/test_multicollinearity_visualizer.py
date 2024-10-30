import unittest
import pandas as pd
import os
from statistics_creator.visualizers.multicollinearity_visualizer import MulticollinearityVisualizer

class TestMulticollinearityVisualizer(unittest.TestCase):
    def setUp(self):
        self.visualizer = MulticollinearityVisualizer()
        self.test_data = pd.DataFrame({
            'Feature': ['A', 'B', 'C'],
            'VIF': [1.5, 2.0, 3.5]
        })
        self.output_path = 'test_output'
        os.makedirs(self.output_path, exist_ok=True)

    def test_visualize(self):
        self.visualizer.visualize(self.test_data, self.output_path)
        self.assertTrue(os.path.exists(os.path.join(self.output_path, 'multicollinearity_vif.png')))

    def tearDown(self):
        for file in os.listdir(self.output_path):
            os.remove(os.path.join(self.output_path, file))
        os.rmdir(self.output_path)

if __name__ == '__main__':
    unittest.main()