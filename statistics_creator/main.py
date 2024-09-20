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

def create_analyzers_and_visualizers() -> Tuple[List[BaseAnalyzer], List[BaseVisualizer]]:
    """
    Create and return lists of analyzers and visualizers.

    Returns:
        Tuple[List[BaseAnalyzer], List[BaseVisualizer]]: A tuple containing a list of analyzers and a list of visualizers.
    """
    analyzers = [
        MissingDataAnalyzer(),
        CorrelationAnalyzer(),
        SummaryStatisticsAnalyzer(),
        ClassAnalyzer(data_config.TARGET_COLUMN),
        FeatureImportanceAnalyzer(data_config.TARGET_COLUMN),
        MulticollinearityAnalyzer(),
        NumericalDistributionAnalyzer(),
        OutlierAnalyzer()
    ]
    visualizers = [
        MissingDataVisualizer(),
        CorrelationVisualizer(),
        SummaryStatisticsVisualizer(),
        ClassVisualizer(),
        FeatureImportanceVisualizer(),
        MulticollinearityVisualizer(),
        NumericalDistributionVisualizer(),
        OutlierVisualizer()
    ]
    return analyzers, visualizers

def main() -> None:
    """
    Main function to run the statistics creation process.

    This function initializes the data loader, creates analyzers and visualizers,
    runs the analysis, and saves the results.
    """
    data_loader = DataLoader()
    analyzers, visualizers = create_analyzers_and_visualizers()

    statistics_creator = StatisticsCreator(data_loader, analyzers, visualizers)
    results = statistics_creator.run_analysis(data_config.PATH)

    logger.info(f"Analysis completed successfully. Summary: {summarize_results(results)}")
    save_results(results, statistics_creator.statistics_folder)

if __name__ == "__main__":
    main()