from typing import List, Any, Tuple
from data_loader import DataLoader
from analyzers.base_analyzer import BaseAnalyzer
from visualizers.base_visualizer import BaseVisualizer
from logger import log_execution_time
from config import data_config

class StatisticsCreator:
    """
    A class to create statistics by running analyzers and visualizers on data.

    This class orchestrates the process of loading data, running various analyzers on it,
    and creating visualizations based on the analysis results.

    Attributes:
        data_loader (DataLoader): An instance of DataLoader for loading the data.
        analyzer_visualizer_pairs (List[Tuple[BaseAnalyzer, BaseVisualizer]]): A list of analyzer-visualizer pairs.
        statistics_folder (str): The folder where statistics and visualizations will be saved.
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
        self.statistics_folder = self.data_loader.create_statistics_folder(data_config.DEFAULT_STATISTICS_FOLDER)

    @log_execution_time
    def run_analysis(self, data_path: str) -> dict[str, Any]:
        """
        Run the analysis on the data and create visualizations.

        It loads the data, applies each analyzer to the data, creates visualizations
        based on the analysis results, and returns a dictionary of all results.

        Args:
            data_path (str): The path to the data file.

        Returns:
            dict[str, Any]: A dictionary containing the results of all analyzers, where the keys
                            are the names of the analyzer classes and the values are the analysis results.

        Note:
            The visualizations are saved in the statistics_folder specified during initialization.
        """
        df = self.data_loader.load_data(data_path)
        results = {}

        for analyzer, visualizer in self.analyzer_visualizer_pairs:
            result = analyzer.analyze(df)
            results[analyzer.__class__.__name__] = result
            visualizer.visualize(result, self.statistics_folder)

        return results