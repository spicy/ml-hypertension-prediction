import pandas as pd

class DataCombiner:
    def __init__(self, csv_files):
        self.csv_files = csv_files
        self.combined_df = pd.DataFrame()

    def combine_data(self):
        for file in self.csv_files:
            df = pd.read_csv(file)

            if self.combined_df.empty:
                self.combined_df = df
            else:
                self.combined_df = pd.merge(self.combined_df, df, on='SEQN', how='outer')

    def save_combined_data(self, output_file='CombinedData.csv'):
        self.combined_df.to_csv(output_file, index=False)
        print(f"Data has been combined and saved to {output_file}")

if __name__ == "__main__":
    files_to_combine = [
        'data/2017-2020/AlcoholUse.csv',
        'data/2017-2020/BloodPressureAndCholesterol.csv',
        'data/2017-2020/DietBehaviorAndNutrition.csv',
        'data/2017-2020/PhysicalActivity.csv',
        'data/2017-2020/Smoking.csv'
    ]
    combiner = DataCombiner(files_to_combine)
    combiner.combine_data()
    combiner.save_combined_data()