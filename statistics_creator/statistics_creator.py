from typing import List, Dict, Any
from data_loader import DataLoader
from analyzers.base_analyzer import BaseAnalyzer
from visualizers.base_visualizer import BaseVisualizer
from logger import log_execution_time

class StatisticsCreator:
    """
    A class to create statistics by running analyzers and visualizers on data.
    """

    def __init__(self, data_loader: DataLoader, analyzers: List[BaseAnalyzer], visualizers: List[BaseVisualizer]):
        """
        Initialize the StatisticsCreator with data loader, analyzers, and visualizers.

        Args:
            data_loader (DataLoader): An instance of DataLoader.
            analyzers (List[BaseAnalyzer]): A list of analyzer instances.
            visualizers (List[BaseVisualizer]): A list of visualizer instances.
        """
        self.data_loader = data_loader
        self.analyzers = analyzers
        self.visualizers = visualizers
        self.statistics_folder = self.data_loader.create_statistics_folder()

    @log_execution_time
    def run_analysis(self, data_path: str) -> Dict[str, Any]:
        """
        Run the analysis on the data and create visualizations.

        Args:
            data_path (str): The path to the data file.

        Returns:
            Dict[str, Any]: A dictionary containing the results of all analyzers.
        """
        df = self.data_loader.load_data(data_path)
        results = {}

        for analyzer in self.analyzers:
            result = analyzer.analyze(df)
            results[analyzer.__class__.__name__] = result

        for visualizer in self.visualizers:
            visualizer.visualize(results[visualizer.__class__.__name__.replace("Visualizer", "Analyzer")], self.statistics_folder)

        return results