import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

# Constants
DATA_DIR = Path("data/processed")
QUESTIONS_DIR = Path("questions")
AUTOFILLED_DATA_FILENAME = "AutofilledData.csv"


# Constants for special tokens
class AutofillToken(Enum):
    VALUE = "##VALUE"  # Copy the original answer value


class TokenProcessor:
    """Handles the processing of special autofill tokens."""

    @staticmethod
    def process_value(value: str) -> str:
        """Direct value copy."""
        return value

    @classmethod
    def get_processor(
        cls, token: AutofillToken | str
    ) -> Optional[Callable[[str], str]]:
        """Get the appropriate processor function for a token."""
        processors = {AutofillToken.VALUE: cls.process_value}
        # Handle both enum and string cases for backward compatibility
        if isinstance(token, str):
            token = next((t for t in AutofillToken if t.value == token), token)
        return processors.get(token)


class DataAutofiller:
    """A class for applying autofill rules to survey data based on question definitions."""

    def __init__(self, input_file: Path):
        """Initialize the DataAutofiller with an input file."""
        self.input_file = input_file
        self.data_df: Optional[pd.DataFrame] = None
        self.questions_data: Dict = {}
        self.autofill_counts: Dict[str, int] = {}
        self._load_questions_data()

    def _load_questions_data(self) -> None:
        """Load all question definitions from JSON files in the questions directory."""
        if not QUESTIONS_DIR.exists():
            raise FileNotFoundError(f"Questions directory not found: {QUESTIONS_DIR}")

        # Process JSON files sequentially
        for json_file in QUESTIONS_DIR.glob("*.json"):
            if result := self._load_single_json(json_file):
                self.questions_data.update(result)

        if not self.questions_data:
            raise ValueError("No question definitions loaded")
        logging.info(f"Loaded {len(self.questions_data)} question definitions")

    @staticmethod
    def _load_single_json(json_file: Path) -> Dict:
        """Load a single JSON file."""
        try:
            with json_file.open("r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON file: {json_file}")
        except Exception as e:
            logging.error(f"Error processing file {json_file}: {str(e)}")
        return {}

    def load_data(self) -> None:
        """Load the input CSV data into a DataFrame."""
        try:
            self.data_df = pd.read_csv(self.input_file)
        except Exception as e:
            logging.error(f"Error reading file {self.input_file}: {str(e)}")
            raise

    def process_autofill_rules(self, question_id: str, answer: str) -> Dict[str, str]:
        """Process autofill rules for a given question and answer."""
        if not (
            question_mappings := self.questions_data.get(question_id, {}).get(
                "mappings", {}
            )
        ):
            return {}

        # Store the current answer for use in _get_autofill_values
        self.current_answer = answer

        try:
            return self._process_numeric_answer(answer, question_mappings)
        except ValueError:
            return self._process_string_answer(answer, question_mappings)

    def _process_numeric_answer(self, answer: str, mappings: Dict) -> Dict[str, str]:
        """Process numeric answers and ranges."""
        answer_num = float(answer)

        for mapping_key, mapping_value in mappings.items():
            if "-" in str(mapping_key):
                range_start, range_end = map(float, mapping_key.split("-"))
                if range_start <= answer_num <= range_end:
                    return self._get_autofill_values(mapping_value)
            elif str(answer_num) == str(mapping_key):
                return self._get_autofill_values(mapping_value)

        return {}

    def _process_string_answer(self, answer: str, mappings: Dict) -> Dict[str, str]:
        """Process string answers."""
        if mapping_value := mappings.get(str(answer)):
            return self._get_autofill_values(mapping_value)
        return {}

    def _get_autofill_values(self, mapping_value: Dict) -> Dict[str, str]:
        """Extract autofill values from mapping, handling special tokens."""
        autofill_dict = mapping_value.get("skip", {}).get("auto_fill", {})

        # Process special tokens in autofill values
        processed_dict = {}
        for key, value in autofill_dict.items():
            # Check if the value matches any AutofillToken value
            if isinstance(value, str) and value == AutofillToken.VALUE.value:
                value = AutofillToken.VALUE

            if processor := TokenProcessor.get_processor(value):
                try:
                    processed_dict[key] = processor(self.current_answer)
                except (ValueError, TypeError) as e:
                    logging.warning(
                        f"Error processing token {value} for {key}: {str(e)}"
                    )
                    processed_dict[key] = value
            else:
                processed_dict[key] = value

        return processed_dict

    def apply_autofill_rules(self) -> None:
        """Apply autofill rules to the entire dataset."""
        if self.data_df is None:
            raise ValueError("Data must be loaded before applying autofill rules")

        logging.info(f"Starting autofill process for {len(self.data_df)} records")
        # Process questions in chunks for better performance
        chunk_size = 1000
        for question_id in self.questions_data:
            if question_id in self.data_df.columns:
                for start_idx in range(0, len(self.data_df), chunk_size):
                    chunk = self.data_df.iloc[start_idx : start_idx + chunk_size]
                    self._process_chunk(question_id, chunk, start_idx)

        # Log summary after processing
        total_autofills = sum(self.autofill_counts.values())
        logging.info(f"Completed autofill process. Total autofills: {total_autofills}")
        for question, count in self.autofill_counts.items():
            logging.info(f"** {question}: {count} autofills")

    def _process_chunk(
        self, question_id: str, chunk: pd.DataFrame, start_idx: int
    ) -> None:
        """Process a chunk of data for autofilling."""
        chunk_autofills = 0
        for idx, answer in chunk[question_id].items():
            if pd.notna(answer):
                if autofill_values := self.process_autofill_rules(
                    question_id, str(answer)
                ):
                    for fill_question, fill_value in autofill_values.items():
                        self.data_df.at[idx, fill_question] = fill_value
                        # Track autofill counts
                        self.autofill_counts[fill_question] = (
                            self.autofill_counts.get(fill_question, 0) + 1
                        )
                        chunk_autofills += 1

        if chunk_autofills > 0:
            logging.debug(
                f"Processed chunk starting at {start_idx} for question {question_id}: "
                f"{chunk_autofills} autofills"
            )

    def save_data(self, filename: str = AUTOFILLED_DATA_FILENAME) -> None:
        """Save the autofilled data to a CSV file."""
        if self.data_df is None or self.data_df.empty:
            raise ValueError("No data to save")

        output_path = DATA_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.data_df.to_csv(output_path, index=False)
            logging.info(f"Autofilled data has been saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving data to {output_path}: {str(e)}")


def main():
    """Main function to execute the data autofilling process for all filtered data files."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Process all filtered data files
    for input_file in DATA_DIR.glob("FilteredCombinedData_*.csv"):
        try:
            year_suffix = input_file.stem.split("_")[1]
            output_filename = AUTOFILLED_DATA_FILENAME.replace(
                ".csv", f"_{year_suffix}.csv"
            )

            logging.info(f"Processing autofill rules for {input_file.name}")
            autofiller = DataAutofiller(input_file)
            autofiller.load_data()
            autofiller.apply_autofill_rules()
            autofiller.save_data(output_filename)

        except Exception as e:
            logging.exception(
                f"An error occurred processing {input_file.name}: {str(e)}"
            )


if __name__ == "__main__":
    main()
