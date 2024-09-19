import unittest
from unittest.mock import Mock, patch
from statistics_creator.main import StatisticsCreator, create_analyzers_and_visualizers
from statistics_creator.data_loader import DataLoader
from statistics_creator.analyzers import MissingDataAnalyzer, CorrelationAnalyzer, SummaryStatisticsAnalyzer, ClassDistributionAnalyzer
from statistics_creator.visualizers import MissingDataVisualizer, CorrelationVisualizer, SummaryStatisticsVisualizer, ClassDistributionVisualizer
import pandas as pd

class TestStatisticsCreator(unittest.TestCase):
    def setUp(self):
        self.data_loader = Mock(spec=DataLoader)
        self.analyzers, self.visualizers = create_analyzers_and_visualizers()
        self.statistics_creator = StatisticsCreator(self.data_loader, self.analyzers, self.visualizers)

    def test_initialization(self):
        self.assertIsInstance(self.statistics_creator.data_loader, DataLoader)
        self.assertEqual(len(self.statistics_creator.analyzers), 4)
        self.assertEqual(len(self.statistics_creator.visualizers), 4)

    @patch('pandas.DataFrame')
    def test_run_analysis(self, mock_df):
        mock_data_path = 'mock/data/path.csv'
        self.data_loader.load_data.return_value = mock_df

        results = self.statistics_creator.run_analysis(mock_data_path)

        self.data_loader.load_data.assert_called_once_with(mock_data_path)
        self.assertEqual(len(results), 4)
        for analyzer in self.analyzers:
            self.assertIn(analyzer.__class__.__name__, results)

class TestCreateAnalyzersAndVisualizers(unittest.TestCase):
    def test_create_analyzers_and_visualizers(self):
        analyzers, visualizers = create_analyzers_and_visualizers()

        self.assertEqual(len(analyzers), 4)
        self.assertEqual(len(visualizers), 4)

        # Test analyzers
        self.assertIsInstance(analyzers[0], MissingDataAnalyzer)
        self.assertIsInstance(analyzers[1], CorrelationAnalyzer)
        self.assertIsInstance(analyzers[2], SummaryStatisticsAnalyzer)
        self.assertIsInstance(analyzers[3], ClassDistributionAnalyzer)

        # Test visualizers
        self.assertIsInstance(visualizers[0], MissingDataVisualizer)
        self.assertIsInstance(visualizers[1], CorrelationVisualizer)
        self.assertIsInstance(visualizers[2], SummaryStatisticsVisualizer)
        self.assertIsInstance(visualizers[3], ClassDistributionVisualizer)

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

if __name__ == '__main__':
    unittest.main()