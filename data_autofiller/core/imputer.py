import pandas as pd
from sklearn.impute import KNNImputer

from ..core.autofiller import AutofillConfig
from ..logger import logger
from ..utils.file_utils import get_data_files


class Imputer:
    def __init__(self, config: AutofillConfig):
        self.config = config
        self.input_files = get_data_files(
            self.config.output_dir, "AutoFilled_Data_*.csv"
        )

    def _drop_and_sort(self, data: pd.DataFrame):
        """Drop all rows from the data that would cause discrepancies in sorting then sort dataframe"""

        # Drop rows with any missing values in systolic, diastolic, or hypertension columns
        data.dropna(axis=0, subset=["BPXOSYAVG", "BPXODIAVG", "HYPERTENSION"])

        # Sort by systolic, diastolic, and age
        data_sys = data.sort_values(by=["BPXOSYAVG", "RIDAGEYR", "BPXODIAVG"])
        data_dia = data.sort_values(by=["BPXODIAVG", "RIDAGEYR", "BPXOSYAVG"])
        data_age = data.sort_values(by=["RIDAGEYR", "BPXOSYAVG", "BPXODIAVG"])

        return [data_sys, data_dia, data_age]

    def _impute(self, data: pd.DataFrame):
        """Impute missing values for singular dataframe using KNNImpute"""
        # Count missing values before imputation
        missing_before = data.isna().sum().sum()

        imputer = KNNImputer(n_neighbors=10)
        imputed_data = imputer.fit_transform(data)
        imputed_df = round(pd.DataFrame(imputed_data, columns=data.columns), 0)

        # Count missing values after imputation
        missing_after = imputed_df.isna().sum().sum()
        imputed_count = missing_before - missing_after
        logger.info(f"Imputed {imputed_count} missing values")

        return imputed_df

    def _average_imputed_dfs(self, imputed_dfs: list) -> pd.DataFrame:
        """Calculate averages for the different sorted dataframes"""
        idx_sorted_dfs = [
            df.sort_values(by="SEQN", ignore_index=True) for df in imputed_dfs
        ]
        sum_df = idx_sorted_dfs[0]
        for i in range(1, len(idx_sorted_dfs)):
            temp = sum_df
            sum_df = temp.add(idx_sorted_dfs[i])
        avg_df = sum_df / 3

        return avg_df

    def impute(self):
        """Impute missing values for all autofilled data"""
        for file in self.input_files:
            logger.info(f"Processing file: {file}")
            data = pd.read_csv(file)

            # Count initial missing values
            initial_missing = data.isna().sum().sum()
            logger.info(f"Initial missing values: {initial_missing}")

            sorted_dfs = self._drop_and_sort(data)
            imputed_dfs = [self._impute(df) for df in sorted_dfs]
            imputed_data = self._average_imputed_dfs(imputed_dfs)

            # Drop systolic and diastolic because they were only left in to calculate HYPERTENSION
            imputed_data = imputed_data.drop(["BPXOSYAVG", "BPXODIAVG"], axis=1)

            # Final check for missing values
            final_missing = imputed_data.isna().sum().sum()
            logger.info(f"Final missing values: {final_missing}")
            logger.info(f"Total values imputed: {initial_missing - final_missing}")

            # Save imputed data
            imputed_data.to_csv(file, index=False)
