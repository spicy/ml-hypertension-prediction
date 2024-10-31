import unittest
from unittest.mock import Mock, patch
from statistics_creator.main import StatisticsCreator, create_analyzers_and_visualizers
from statistics_creator.data_loader import DataLoader
from statistics_creator.analyzers.missing_data_analyzer import MissingDataAnalyzer
from statistics_creator.analyzers.correlation_analyzer import CorrelationAnalyzer
from statistics_creator.analyzers.summary_statistics_analyzer import SummaryStatisticsAnalyzer
from statistics_creator.analyzers.class_analyzer import ClassAnalyzer
from statistics_creator.analyzers.numerical_distribution_analyzer import NumericalDistributionAnalyzer
from statistics_creator.analyzers.multicollinearity_analyzer import MulticollinearityAnalyzer
from statistics_creator.visualizers.missing_data_visualizer import MissingDataVisualizer
from statistics_creator.visualizers.correlation_visualizer import CorrelationVisualizer
from statistics_creator.visualizers.summary_statistics_visualizer import SummaryStatisticsVisualizer
from statistics_creator.visualizers.class_visualizer import ClassVisualizer
from statistics_creator.visualizers.numerical_distribution_visualizer import NumericalDistributionVisualizer
from statistics_creator.visualizers.feature_importance_visualizer import FeatureImportanceVisualizer
import pandas as pd
import numpy as np

class TestStatisticsCreator(unittest.TestCase):
    def setUp(self):
        self.data_loader = Mock(spec=DataLoader)
        self.analyzers, self.visualizers = create_analyzers_and_visualizers()
        self.statistics_creator = StatisticsCreator(self.data_loader, self.analyzers, self.visualizers)

    def test_initialization(self):
        self.assertIsInstance(self.statistics_creator.data_loader, DataLoader)
        self.assertEqual(len(self.statistics_creator.analyzers), 6)
        self.assertEqual(len(self.statistics_creator.visualizers), 6)

    @patch('pandas.DataFrame')
    def test_run_analysis(self, mock_df):
        mock_data_path = 'mock/data/path.csv'
        self.data_loader.load_data.return_value = mock_df

        results = self.statistics_creator.run_analysis(mock_data_path)

        self.data_loader.load_data.assert_called_once_with(mock_data_path)
        self.assertEqual(len(results), 6)
        for analyzer in self.analyzers:
            self.assertIn(analyzer.__class__.__name__, results)

class TestCreateAnalyzersAndVisualizers(unittest.TestCase):
    def test_create_analyzers_and_visualizers(self):
        analyzers, visualizers = create_analyzers_and_visualizers()

        self.assertEqual(len(analyzers), 6)
        self.assertEqual(len(visualizers), 6)

        # Test analyzers
        self.assertIsInstance(analyzers[0], MissingDataAnalyzer)
        self.assertIsInstance(analyzers[1], CorrelationAnalyzer)
        self.assertIsInstance(analyzers[2], SummaryStatisticsAnalyzer)
        self.assertIsInstance(analyzers[3], ClassAnalyzer)
        self.assertIsInstance(analyzers[4], NumericalDistributionAnalyzer)
        self.assertIsInstance(analyzers[5], MulticollinearityAnalyzer)

        # Test visualizers
        self.assertIsInstance(visualizers[0], MissingDataVisualizer)
        self.assertIsInstance(visualizers[1], CorrelationVisualizer)
        self.assertIsInstance(visualizers[2], SummaryStatisticsVisualizer)
        self.assertIsInstance(visualizers[3], ClassVisualizer)
        self.assertIsInstance(visualizers[4], NumericalDistributionVisualizer)
        self.assertIsInstance(visualizers[5], FeatureImportanceVisualizer)

class TestAnalyzers(unittest.TestCase):
    @patch('pandas.DataFrame')
    def test_missing_data_analyzer(self, mock_df):
        analyzer = MissingDataAnalyzer()
        result = analyzer.analyze(mock_df)
        self.assertIsInstance(result, pd.Series)

    @patch('pandas.DataFrame')
    def test_correlation_analyzer(self, mock_df):
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(mock_df)
        self.assertIsInstance(result, pd.DataFrame)

    @patch('pandas.DataFrame')
    def test_summary_statistics_analyzer(self, mock_df):
        analyzer = SummaryStatisticsAnalyzer()
        result = analyzer.analyze(mock_df)
        self.assertIsInstance(result, pd.DataFrame)

    @patch('pandas.DataFrame')
    def test_class_distribution_analyzer(self, mock_df):
        analyzer = ClassDistributionAnalyzer('target_column')
        mock_df.columns = ['target_column']
        mock_df['target_column'] = [1, 2, 1, 3, 2, 1]
        result = analyzer.analyze(mock_df)
        self.assertIsInstance(result, pd.Series)

    @patch('pandas.DataFrame')
    def test_numerical_distribution_analyzer(self, mock_df):
        analyzer = NumericalDistributionAnalyzer()
        mock_df.select_dtypes.return_value.columns = ['col1', 'col2']
        mock_df['col1'] = pd.Series([1, 2, 3, 4, 5])
        mock_df['col2'] = pd.Series([2, 4, 6, 8, 10])
        result = analyzer.analyze(mock_df)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        for col_stats in result.values():
            self.assertIsInstance(col_stats, dict)
            self.assertTrue(all(key in col_stats for key in ['mean', 'median', 'std', 'skewness', 'kurtosis', 'min', 'max', 'q1', 'q3']))

    @patch('pandas.DataFrame')
    def test_multicollinearity_analyzer(self, mock_df):
        analyzer = MulticollinearityAnalyzer()
        mock_df.select_dtypes.return_value = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        result = analyzer.analyze(mock_df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ['Feature', 'VIF'])

class TestVisualizers(unittest.TestCase):
    @patch('matplotlib.pyplot.savefig')
    def test_missing_data_visualizer(self, mock_savefig):
        visualizer = MissingDataVisualizer()
        data = pd.Series({'A': 10, 'B': 20, 'C': 30})
        visualizer.visualize(data, 'output_path')
        mock_savefig.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    def test_correlation_visualizer(self, mock_savefig):
        visualizer = CorrelationVisualizer()
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        visualizer.visualize(data, 'output_path')
        mock_savefig.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    def test_summary_statistics_visualizer(self, mock_savefig):
        visualizer = SummaryStatisticsVisualizer()
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}).describe()
        visualizer.visualize(data, 'output_path')
        mock_savefig.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    def test_class_distribution_visualizer(self, mock_savefig):
        visualizer = ClassDistributionVisualizer()
        data = pd.Series({'Class1': 40, 'Class2': 60})
        visualizer.visualize(data, 'output_path')
        mock_savefig.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    def test_numerical_distribution_visualizer(self, mock_savefig):
        visualizer = NumericalDistributionVisualizer()
        data = {
            'col1': {'mean': 3, 'median': 3, 'std': 1.4, 'skewness': 0, 'kurtosis': -1.2, 'min': 1, 'max': 5, 'q1': 2, 'q3': 4},
            'col2': {'mean': 6, 'median': 6, 'std': 2.8, 'skewness': 0, 'kurtosis': -1.2, 'min': 2, 'max': 10, 'q1': 4, 'q3': 8}
        }
        visualizer.visualize(data, 'output_path')
        self.assertEqual(mock_savefig.call_count, 2)

    @patch('matplotlib.pyplot.savefig')
    def test_feature_importance_visualizer(self, mock_savefig):
        visualizer = FeatureImportanceVisualizer()
        data = pd.Series({'Feature1': 0.5, 'Feature2': 0.3, 'Feature3': 0.2})
        visualizer.visualize(data, 'output_path')
        mock_savefig.assert_called_once()

if __name__ == '__main__':
    unittest.main()