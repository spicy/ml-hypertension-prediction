import pandas as pd
import os
from pathlib import Path
from typing import List, Optional

class DataCombiner:
    OUTPUT_DIR = Path('data') / '2017-2020' / 'processed'

    def __init__(self, input_files: List[str]):
        self.input_files = input_files
        self.combined_df: Optional[pd.DataFrame] = None
        self.filtered_df: Optional[pd.DataFrame] = None

    def combine_data(self) -> None:
        dfs = [pd.read_csv(file) for file in self.input_files]
        self.combined_df = pd.concat(dfs, ignore_index=True)

    def filter_data(self, data_filter: 'DataFilter') -> None:
        if self.combined_df is None:
            raise ValueError("Data must be combined before filtering.")
        self.filtered_df = data_filter.apply(self.combined_df)

    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        output_path = self.OUTPUT_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        print(f"Data has been saved to {output_path}")

    def save_combined_data(self, filename: str = 'UnfilteredCombinedData.csv') -> None:
        if self.combined_df is None:
            raise ValueError("No combined data to save.")
        self.save_data(self.combined_df, filename)

    def save_filtered_data(self, filename: str = 'FilteredCombinedData.csv') -> None:
        if self.filtered_df is None:
            raise ValueError("No filtered data to save.")
        self.save_data(self.filtered_df, filename)

class DataFilter:
    def __init__(self, relevant_columns: List[str]):
        self.relevant_columns = relevant_columns

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.relevant_columns]

def main():
    files_to_combine = [
        'data/2017-2020/raw/AlcoholUse.csv',
        'data/2017-2020/raw/BloodPressureAndCholesterol.csv',
        'data/2017-2020/raw/DietBehaviorAndNutrition.csv',
        'data/2017-2020/raw/PhysicalActivity.csv',
        'data/2017-2020/raw/Smoking.csv'
    ]

    # List of columns to include in the filtered combined dataset
    relevant_columns = [
        "SEQN",  # Respondent sequence number
        # Alcohol consumption
        "ALQ111", "ALQ121", "ALQ130", "ALQ142", "ALQ270", "ALQ280", "ALQ290", "ALQ151", "ALQ170",
        # Blood pressure and cholesterol
        "BPQ020", "BPQ030", "BPQ040A", "BPQ050A", "BPQ080", "BPQ060", "BPQ070", "BPQ090D", "BPQ100D",
        # Physical activity
        "PAQ605", "PAQ610", "PAD615", "PAQ620", "PAQ625", "PAD630", "PAQ635", "PAQ640", "PAD645",
        "PAQ650", "PAQ655", "PAD660", "PAQ665", "PAQ670", "PAD675", "PAD680",
        # Smoking
        "SMQ020", "SMD030", "SMQ040", "SMQ050Q", "SMQ050U", "SMD057", "SMQ078", "SMD641", "SMD650",
        "SMD100FL", "SMD100MN", "SMQ670", "SMQ621", "SMD630",
        # Diet
        "DBQ700"
    ]

    combiner = DataCombiner(files_to_combine)
    combiner.combine_data()
    combiner.save_combined_data()

    data_filter = DataFilter(relevant_columns)
    combiner.filter_data(data_filter)
    combiner.save_filtered_data()

if __name__ == "__main__":
    main()