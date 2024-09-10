from data_loader import create_statistics_folder, load_data
from data_analyzer import analyze_missing_data, calculate_correlation_matrix, generate_summary_statistics, calculate_class_distribution
from data_visualizer import plot_missing_data, plot_correlation_matrix
from utils import save_missing_data_summary, save_summary_statistics, save_class_distribution, save_target_correlations
from constants import DATA_PATH, TARGET_COLUMN
from logger import logger, log_execution_time

@log_execution_time
def main():
    logger.info("Starting data analysis process...")
    statistics_folder = create_statistics_folder()
    df = load_data(DATA_PATH)

    logger.info("Analyzing missing data...")
    missing_percentage_sorted = analyze_missing_data(df)
    plot_missing_data(missing_percentage_sorted, statistics_folder)
    save_missing_data_summary(missing_percentage_sorted, statistics_folder)

    logger.info("Calculating correlation matrix...")
    correlation_matrix = calculate_correlation_matrix(df)
    plot_correlation_matrix(correlation_matrix, statistics_folder)
    save_target_correlations(correlation_matrix, TARGET_COLUMN, statistics_folder)

    logger.info("Generating summary statistics...")
    summary_statistics = generate_summary_statistics(df)
    save_summary_statistics(summary_statistics, statistics_folder)

    logger.info("Calculating class distribution...")
    class_distribution = calculate_class_distribution(df, TARGET_COLUMN)
    save_class_distribution(class_distribution, statistics_folder)

    logger.info("Data analysis process completed successfully.")

if __name__ == "__main__":
    main()
