import unittest
import pandas as pd
from analyzers.summary_statistics_analyzer import SummaryStatisticsAnalyzer

class TestSummaryStatisticsAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = SummaryStatisticsAnalyzer()
        self.test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

    def test_analyze(self):
        result = self.analyzer.analyze(self.test_data)
        expected = pd.DataFrame({
            'A': [5.0, 3.0, 1.581139, 1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [5.0, 30.0, 15.811388, 10.0, 20.0, 30.0, 40.0, 50.0]
        }, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        pd.testing.assert_frame_equal(result, expected, check_less_precise=2)

    def test_analyze_single_column(self):
        data = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        result = self.analyzer.analyze(data)
        self.assertEqual(result.shape, (8, 1))

    def test_analyze_empty_dataframe(self):
        data = pd.DataFrame()
        result = self.analyzer.analyze(data)
        self.assertTrue(result.empty)

if __name__ == '__main__':
    unittest.main()