from typing import Any, List, Tuple

from analyzers.base_analyzer import BaseAnalyzer
from config import data_config
from data_loader import DataLoader
from logger import log_execution_time
from visualizers.base_visualizer import BaseVisualizer


class StatisticsCreator:
    """
    A class to create statistics by running analyzers and visualizers on data.

    This class orchestrates the process of loading data, running various analyzers on it,
    and creating visualizations based on the analysis results.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        analyzer_visualizer_pairs: List[Tuple[BaseAnalyzer, BaseVisualizer]],
    ):
        """
        Initialize the StatisticsCreator with data loader, analyzers, and visualizers.
        """
        self.data_loader = data_loader
        self.analyzer_visualizer_pairs = analyzer_visualizer_pairs
        self._statistics_folder = None

    @property
    def statistics_folder(self) -> str:
        if self._statistics_folder is None:
            self._statistics_folder = self.data_loader.create_statistics_folder(
                data_config.DEFAULT_STATISTICS_FOLDER
            )
        return self._statistics_folder

    @statistics_folder.setter
    def statistics_folder(self, value: str):
        self._statistics_folder = value

    @log_execution_time
    def run_analysis(self, data_path: str) -> dict[str, Any]:
        """
        Run the analysis on the data and create visualizations.

        It loads the data, applies each analyzer to the data, creates visualizations
        based on the analysis results, and returns a dictionary of all results.
        """
        df = self.data_loader.load_data(data_path)
        results = {}

        for analyzer, visualizer in self.analyzer_visualizer_pairs:
            result = analyzer.analyze(df)
            results[analyzer.__class__.__name__] = result
            visualizer.visualize(result, self.statistics_folder)

        return results
