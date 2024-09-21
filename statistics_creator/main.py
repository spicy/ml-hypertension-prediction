from typing import List, Tuple
from data_loader import DataLoader
from analyzers.base_analyzer import BaseAnalyzer
from visualizers.base_visualizer import BaseVisualizer
from analyzers import MissingDataAnalyzer, CorrelationAnalyzer, SummaryStatisticsAnalyzer, ClassAnalyzer, FeatureImportanceAnalyzer, MulticollinearityAnalyzer, NumericalDistributionAnalyzer, OutlierAnalyzer
from visualizers import MissingDataVisualizer, CorrelationVisualizer, SummaryStatisticsVisualizer, ClassVisualizer, FeatureImportanceVisualizer, MulticollinearityVisualizer, NumericalDistributionVisualizer, OutlierVisualizer
from config import data_config
from logger import logger
from statistics_creator import StatisticsCreator
from utils import summarize_results, save_results
import os

def create_analyzer_visualizer_pairs() -> List[Tuple[BaseAnalyzer, BaseVisualizer]]:
    """
    Create and return a list of analyzer-visualizer pairs.

    Returns:
        List[Tuple[BaseAnalyzer, BaseVisualizer]]: A list of tuples, each containing an analyzer and its corresponding visualizer.
    """
    return [
        (MissingDataAnalyzer(), MissingDataVisualizer()),
        (CorrelationAnalyzer(), CorrelationVisualizer()),
        (SummaryStatisticsAnalyzer(), SummaryStatisticsVisualizer()),
        (ClassAnalyzer(data_config.TARGET_COLUMN), ClassVisualizer()),
        (FeatureImportanceAnalyzer(data_config.TARGET_COLUMN), FeatureImportanceVisualizer()),
        (MulticollinearityAnalyzer(), MulticollinearityVisualizer()),
        (NumericalDistributionAnalyzer(), NumericalDistributionVisualizer()),
        (OutlierAnalyzer(), OutlierVisualizer())
    ]

def main() -> None:
    """
    Main function to run the statistics creation process.

    This function initializes the data loader, creates analyzer-visualizer pairs,
    runs the analysis, and saves the results.
    """
    data_loader = DataLoader()
    analyzer_visualizer_pairs = create_analyzer_visualizer_pairs()

    statistics_creator = StatisticsCreator(data_loader, analyzer_visualizer_pairs)
    results = statistics_creator.run_analysis(data_config.PATH)

    logger.info(f"Analysis completed successfully. Summary: {summarize_results(results)}")
    save_results(results, statistics_creator.statistics_folder)

if __name__ == "__main__":
    main()