import sys
from os.path import abspath, dirname
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(dirname(dirname(abspath(__file__))))

from data_combiner.config import config
from data_combiner.core.combiner import DataCombiner
from data_combiner.core.filter import DataFilter
from data_combiner.logger import logger
from data_combiner.utils.file_utils import get_data_directories, get_relevant_columns


def main():
    """Main function to execute the data combining and filtering process."""
    logger.info("Starting data combination process...")

    try:
        data_dirs = get_data_directories()
        logger.info(f"Found {len(data_dirs)} data directories to process")

        if not data_dirs:
            logger.warning("No data directories found to process")
            return

        for data_dir in data_dirs:
            if not data_dir.exists():
                logger.warning(f"Year directory not found: {data_dir}")
                continue

            files_to_combine = list(data_dir.glob("*.csv"))
            if not files_to_combine:
                logger.warning(f"No CSV files found in {data_dir}")
                continue

            try:
                year_suffix = f"_{data_dir.name}"
                unfiltered_filename = config.UNFILTERED_DATA_FILENAME.replace(
                    ".csv", f"{year_suffix}.csv"
                )
                filtered_filename = config.FILTERED_DATA_FILENAME.replace(
                    ".csv", f"{year_suffix}.csv"
                )

                logger.info(f"Processing data for {data_dir.name}")
                combiner = DataCombiner(files_to_combine)
                combiner.combine_data()
                combiner.save_combined_data(unfiltered_filename)

                # Validate combined data only if enabled in config
                if config.VALIDATE_DATA:
                    unfiltered_path = config.PROCESSED_DIR / unfiltered_filename
                    combiner.validate_combined_data(unfiltered_path)
                    logger.info("Data validation completed")
                else:
                    logger.info("Data validation skipped (disabled in config)")

                # Filter and save filtered data
                data_filter = DataFilter(get_relevant_columns())
                combiner.filter_data(data_filter)
                combiner.save_filtered_data(filtered_filename)

            except Exception as e:
                logger.exception(f"Error processing {data_dir.name}: {str(e)}")

    except Exception as e:
        logger.exception("An error occurred during the data combination process")


if __name__ == "__main__":
    main()
