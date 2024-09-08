import os
import pandas as pd

def save_missing_data_summary(missing_percentage: pd.Series, statistics_folder: str):
    print("Saving missing data summary...")
    missing_summary = pd.DataFrame({
        'Column': missing_percentage.index,
        'Percentage Missing': missing_percentage.values
    })
    missing_summary = missing_summary.sort_values('Percentage Missing', ascending=False)
    csv_path = os.path.join(statistics_folder, 'missing_data_summary.csv')
    missing_summary.to_csv(csv_path, index=False)
    print(f"Missing data summary saved to: {csv_path}")

def save_summary_statistics(summary_statistics: pd.DataFrame, statistics_folder: str):
    print("Saving summary statistics...")
    csv_path = os.path.join(statistics_folder, 'summary_statistics.csv')
    summary_statistics.to_csv(csv_path)
    print(f"Summary statistics CSV saved to: {csv_path}")

    html_content = summary_statistics.to_html()
    html_path = os.path.join(statistics_folder, 'summary_statistics.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f"Summary statistics HTML saved to: {html_path}")