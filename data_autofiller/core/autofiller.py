import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
import psutil

from ..core.exceptions import AutofillErrorCode, AutofillException
from ..core.interfaces import DataReader, QuestionRepository, RuleEngine
from ..logger import logger
from ..utils.decorators import profile_performance


@dataclass
class AutofillConfig:
    chunk_size: int
    allow_missing_columns: bool
    seqn_column: str
    questions_dir: Path
    data_dir: Path
    output_dir: Path

    def __post_init__(self):
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if not self.questions_dir.exists():
            raise FileNotFoundError(
                f"Questions directory not found: {self.questions_dir}"
            )
        if not any(self.questions_dir.glob("*.json")):
            raise FileNotFoundError(
                f"No JSON files found in questions directory: {self.questions_dir}"
            )
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")


class DataAutofiller:
    def __init__(
        self,
        data_reader: DataReader,
        question_repository: QuestionRepository,
        rule_engine: RuleEngine,
        config: AutofillConfig,
    ):
        self.data_reader = data_reader
        self.question_repository = question_repository
        self.rule_engine = rule_engine
        self.config = config
        self.records_processed = 0
        self.autofill_counts: Dict[str, int] = {}

    def _get_optimal_chunk_size(self, file_size: int) -> int:
        available_memory = psutil.virtual_memory().available
        estimated_row_size = 1000  # Estimate in bytes
        return min(
            max(1000, available_memory // (estimated_row_size * 10)),
            100000,  # Upper limit
        )

    @profile_performance
    def process_file(self, input_file: Path, output_file: Path) -> None:
        """Process file in chunks to minimize memory usage."""
        questions_data = self.question_repository.load_questions(
            self.config.questions_dir
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Get file size for chunk size optimization
        file_size = input_file.stat().st_size
        chunk_size = self._get_optimal_chunk_size(file_size)

        # Optimize dtype usage
        dtype_dict = self._get_optimized_dtypes(input_file)

        for chunk in pd.read_csv(
            input_file,
            chunksize=chunk_size,
            dtype=dtype_dict,
            usecols=self._get_required_columns(questions_data),
        ):
            self._process_chunk_optimized(chunk, output_file, questions_data)

    def _validate_chunk(self, chunk: pd.DataFrame) -> None:
        """Validate the data structure using the first chunk."""
        if (
            self.config.seqn_column not in chunk.columns
            and not self.config.allow_missing_columns
        ):
            raise AutofillException(
                AutofillErrorCode.VALIDATION_ERROR,
                f"Missing required column: {self.config.seqn_column}",
            )

    @profile_performance
    def _process_chunk_optimized(
        self,
        chunk: pd.DataFrame,
        output_file: Path,
        questions_data: Dict,
        write_header: bool,
    ) -> None:
        try:
            start_time = time.time()
            # Optimize memory before processing
            chunk = self._optimize_dataframe_memory(chunk)

            processed_chunk = self.rule_engine.process_chunk(chunk, questions_data)

            # Optimize memory before saving
            processed_chunk = self._optimize_dataframe_memory(processed_chunk)

            processed_chunk.to_csv(
                output_file,
                mode="w" if write_header else "a",
                header=write_header,
                index=False,
                compression="gzip" if output_file.suffix == ".gz" else None,
            )

            self._update_metrics(chunk, processed_chunk, start_time)
        except Exception as e:
            raise AutofillException(
                AutofillErrorCode.PROCESSING_ERROR, f"Error processing chunk: {str(e)}"
            )

    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting numeric types and categorizing strings."""
        for col in df.select_dtypes(include=["int"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        for col in df.select_dtypes(include=["float"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        for col in df.select_dtypes(include=["object"]).columns:
            # If less than 50% unique values
            if df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype("category")
        return df
