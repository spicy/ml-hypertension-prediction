from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..config import config
from ..logger import logger
from .processor import RuleProcessor


class DataAutofiller:
    """A class for applying autofill rules to survey data based on question definitions."""

    def __init__(self, input_file: Path):
        """Initialize the DataAutofiller with an input file."""
        self.input_file = input_file
        self.data_df: Optional[pd.DataFrame] = None
        self.questions_data: Dict = {}
        self.autofill_counts: Dict[str, int] = {}
        self.rule_processor = RuleProcessor()
        self._load_questions_data()

    def _load_questions_data(self) -> None:
        """Load all question definitions from JSON files."""
        if not config.QUESTIONS_DIR.exists():
            raise FileNotFoundError(
                f"Questions directory not found: {config.QUESTIONS_DIR}"
            )

        for json_file in config.QUESTIONS_DIR.glob("*.json"):
            if result := self._load_single_json(json_file):
                self.questions_data.update(result)

        if not self.questions_data:
            raise ValueError("No question definitions loaded")
        logger.info(f"Loaded {len(self.questions_data)} question definitions")

    def load_data(self) -> None:
        """Load the input CSV data into a DataFrame."""
        try:
            self.data_df = pd.read_csv(self.input_file)
        except Exception as e:
            logger.error(f"Error reading file {self.input_file}: {str(e)}")
            raise

    def apply_autofill_rules(self) -> None:
        """Apply autofill rules to the entire dataset."""
        if self.data_df is None:
            raise ValueError("Data must be loaded before applying autofill rules")

        logger.info(f"Starting autofill process for {len(self.data_df)} records")
        chunk_size = 1000
        for question_id in self.questions_data:
            if question_id in self.data_df.columns:
                for start_idx in range(0, len(self.data_df), chunk_size):
                    chunk = self.data_df.iloc[start_idx : start_idx + chunk_size]
                    self._process_chunk(question_id, chunk, start_idx)

        self._log_autofill_summary()

    def save_data(self, filename: str = config.AUTOFILLED_DATA_FILENAME) -> None:
        """Save the autofilled data to a CSV file."""
        if self.data_df is None or self.data_df.empty:
            raise ValueError("No data to save")

        output_path = config.DATA_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.data_df.to_csv(output_path, index=False)
            logger.info(f"Autofilled data has been saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving data to {output_path}: {str(e)}")
