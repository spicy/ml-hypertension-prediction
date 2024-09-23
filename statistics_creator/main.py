from typing import List, Tuple
from data_loader import DataLoader
from analyzers.base_analyzer import BaseAnalyzer
from visualizers.base_visualizer import BaseVisualizer
from analyzers import (
    MissingDataAnalyzer, 
    CorrelationMulticollinearityAnalyzer, 
    ComprehensiveNumericalAnalyzer, 
    ClassAnalyzer, 
    FeatureImportanceAnalyzer, 
    OutlierAnalyzer
)
from visualizers import (
    MissingDataVisualizer, 
    CorrelationMulticollinearityVisualizer, 
    ComprehensiveNumericalVisualizer, 
    ClassVisualizer, 
    FeatureImportanceVisualizer, 
    OutlierVisualizer
)
from config import data_config
from logger import logger
from statistics_creator import StatisticsCreator
from utils import summarize_results, save_results

def create_analyzer_visualizer_pairs() -> List[Tuple[BaseAnalyzer, BaseVisualizer]]:
    """
    Create and return a list of analyzer-visualizer pairs.

    Returns:
        List[Tuple[BaseAnalyzer, BaseVisualizer]]: A list of tuples, each containing an analyzer and its corresponding visualizer.
    """
    return [
        (MissingDataAnalyzer(), MissingDataVisualizer()),
        (CorrelationMulticollinearityAnalyzer(), CorrelationMulticollinearityVisualizer()),
        (ComprehensiveNumericalAnalyzer(), ComprehensiveNumericalVisualizer()),
        (ClassAnalyzer(data_config.TARGET_COLUMN), ClassVisualizer()),
        (FeatureImportanceAnalyzer(data_config.TARGET_COLUMN), FeatureImportanceVisualizer()),
        (OutlierAnalyzer(), OutlierVisualizer())
    ]

def main() -> None:
    """
    Main function to run the statistics creation process.

    It performs the following steps:
    1. Initializes the data loader
    2. Creates analyzer-visualizer pairs
    3. Initializes the StatisticsCreator with the data loader and analyzer-visualizer pairs
    4. Runs the analysis on the data specified in the configuration
    5. Logs a summary of the analysis results
    6. Saves the detailed results to a file
    """
    data_loader = DataLoader()
    analyzer_visualizer_pairs = create_analyzer_visualizer_pairs()

    statistics_creator = StatisticsCreator(data_loader, analyzer_visualizer_pairs)
    results = statistics_creator.run_analysis(data_config.PATH)

    logger.info(f"Analysis completed successfully. Summary: {summarize_results(results)}")
    save_results(results, statistics_creator.statistics_folder)

if __name__ == "__main__":
    main()