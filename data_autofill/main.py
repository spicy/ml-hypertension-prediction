import sys
from os.path import abspath, dirname

# Add the parent directory to sys.path
sys.path.append(dirname(dirname(abspath(__file__))))

from data_autofill.config import config
from data_autofill.core.autofiller import DataAutofiller
from data_autofill.logger import logger


def main():
    """Main function to execute the data autofilling process."""
    logger.info("Starting data autofilling process...")

    try:
        # Process all filtered data files
        for input_file in config.DATA_DIR.glob("FilteredCombinedData_*.csv"):
            try:
                year_suffix = input_file.stem.split("_")[1]
                output_filename = config.AUTOFILLED_DATA_FILENAME.replace(
                    ".csv", f"_{year_suffix}.csv"
                )

                logger.info(f"Processing autofill rules for {input_file.name}")
                autofiller = DataAutofiller(input_file)
                autofiller.load_data()
                autofiller.apply_autofill_rules()
                autofiller.save_data(output_filename)

            except Exception as e:
                logger.exception(
                    f"An error occurred processing {input_file.name}: {str(e)}"
                )

    except Exception as e:
        logger.exception("An error occurred during the data autofilling process")
        sys.exit(1)


if __name__ == "__main__":
    main()
