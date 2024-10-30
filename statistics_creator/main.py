import os
from typing import List, Tuple

from analyzers import (
    ClassAnalyzer,
    ComprehensiveNumericalAnalyzer,
    CorrelationMulticollinearityAnalyzer,
    FeatureImportanceAnalyzer,
    MissingDataAnalyzer,
    OutlierAnalyzer,
)
from analyzers.base_analyzer import BaseAnalyzer
from config import data_config
from data_loader import DataLoader
from logger import logger
from utils import save_results, summarize_results
from visualizers import (
    ClassVisualizer,
    ComprehensiveNumericalVisualizer,
    CorrelationMulticollinearityVisualizer,
    FeatureImportanceVisualizer,
    MissingDataVisualizer,
    OutlierVisualizer,
)
from visualizers.base_visualizer import BaseVisualizer

from statistics_creator import StatisticsCreator


def create_analyzer_visualizer_pairs() -> List[Tuple[BaseAnalyzer, BaseVisualizer]]:
    """
    Create and return a list of analyzer-visualizer pairs.

    Returns:
        List[Tuple[BaseAnalyzer, BaseVisualizer]]: A list of tuples, each containing an analyzer and its corresponding visualizer.
    """
    return [
        (MissingDataAnalyzer(), MissingDataVisualizer()),
        (
            CorrelationMulticollinearityAnalyzer(),
            CorrelationMulticollinearityVisualizer(),
        ),
        (ComprehensiveNumericalAnalyzer(), ComprehensiveNumericalVisualizer()),
        (ClassAnalyzer(data_config.TARGET_COLUMN), ClassVisualizer()),
        (
            FeatureImportanceAnalyzer(data_config.TARGET_COLUMN),
            FeatureImportanceVisualizer(),
        ),
        (OutlierAnalyzer(), OutlierVisualizer()),
    ]


def main() -> None:
    """
    Main function to run the statistics creation process.

    This function orchestrates the entire statistics creation workflow by:
    1. Initializing a DataLoader instance to handle data loading operations
    2. Creating pairs of analyzers and visualizers for different statistical analyses
    3. Setting up a StatisticsCreator with the data loader and analyzer-visualizer pairs
    4. Processing multiple filtered data files from different year ranges
    5. For each data file:
        - Extracts the year range from the filename
        - Creates a year-specific statistics folder
        - Runs statistical analyses and generates visualizations
        - Logs analysis summaries
        - Saves detailed results to files
    6. Handles errors gracefully with appropriate logging
    """
    data_loader = DataLoader()
    analyzer_visualizer_pairs = create_analyzer_visualizer_pairs()
    statistics_creator = StatisticsCreator(data_loader, analyzer_visualizer_pairs)

    # Get all filtered data files
    processed_dir = data_config.PROCESSED_DIR
    filtered_files = list(processed_dir.glob(data_config.FILTERED_DATA_PATTERN))

    if not filtered_files:
        logger.error(f"No filtered data files found in {processed_dir}")
        return

    # Process each filtered data file
    for data_file in filtered_files:
        year_range = data_file.stem.split("_")[-1]  # Extract year range from filename
        logger.info(f"Processing data for year range: {year_range}")

        # Create year-specific statistics folder
        year_stats_folder = os.path.join(
            data_config.DEFAULT_STATISTICS_FOLDER, year_range
        )
        statistics_creator.statistics_folder = data_loader.create_statistics_folder(
            year_stats_folder
        )

        try:
            results = statistics_creator.run_analysis(str(data_file))
            logger.info(
                f"Analysis completed for {year_range}. Summary: {summarize_results(results)}"
            )
            save_results(results, statistics_creator.statistics_folder)
        except Exception as e:
            logger.error(f"Error processing {year_range}: {str(e)}")


if __name__ == "__main__":
    main()
