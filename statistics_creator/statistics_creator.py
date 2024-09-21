from typing import List, Any, Tuple
from data_loader import DataLoader
from analyzers.base_analyzer import BaseAnalyzer
from visualizers.base_visualizer import BaseVisualizer
from logger import log_execution_time

class StatisticsCreator:
    """
    A class to create statistics by running analyzers and visualizers on data.
    """

    def __init__(self, data_loader: DataLoader, analyzer_visualizer_pairs: List[Tuple[BaseAnalyzer, BaseVisualizer]]):
        """
        Initialize the StatisticsCreator with data loader, analyzers, and visualizers.

        Args:
            data_loader (DataLoader): An instance of DataLoader.
            analyzer_visualizer_pairs (List[Tuple[BaseAnalyzer, BaseVisualizer]]): A list of pairs of analyzer and visualizer instances.
        """
        self.data_loader = data_loader
        self.analyzer_visualizer_pairs = analyzer_visualizer_pairs
        self.statistics_folder = self.data_loader.create_statistics_folder()

    @log_execution_time
    def run_analysis(self, data_path: str) -> dict[str, Any]:
        """
        Run the analysis on the data and create visualizations.

        Args:
            data_path (str): The path to the data file.

        Returns:
            dict[str, Any]: A dictionary containing the results of all analyzers.
        """
        df = self.data_loader.load_data(data_path)
        results = {}

        for analyzer, visualizer in self.analyzer_visualizer_pairs:
            result = analyzer.analyze(df)
            results[analyzer.__class__.__name__] = result
            visualizer.visualize(result, self.statistics_folder)

        return results