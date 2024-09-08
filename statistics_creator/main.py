from data_loader import create_statistics_folder, load_data
from data_analyzer import analyze_missing_data, calculate_correlation_matrix, generate_summary_statistics
from data_visualizer import plot_missing_data, plot_correlation_matrix
from utils import save_missing_data_summary, save_summary_statistics
from constants import COLUMN_DEFINITIONS, DATA_PATH

def main():
    print("Starting data analysis process...")
    statistics_folder = create_statistics_folder()
    df = load_data(DATA_PATH)

    missing_percentage_sorted = analyze_missing_data(df)
    plot_missing_data(missing_percentage_sorted, statistics_folder, COLUMN_DEFINITIONS)
    save_missing_data_summary(missing_percentage_sorted, statistics_folder)

    correlation_matrix = calculate_correlation_matrix(df)
    plot_correlation_matrix(correlation_matrix, statistics_folder, COLUMN_DEFINITIONS)

    summary_statistics = generate_summary_statistics(df)
    save_summary_statistics(summary_statistics, statistics_folder)

    print("Data analysis process completed successfully.")

if __name__ == "__main__":
    main()