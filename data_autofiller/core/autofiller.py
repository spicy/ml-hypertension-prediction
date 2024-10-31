import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..core.exceptions import AutofillErrorCode, AutofillException
from ..core.interfaces import DataReader, QuestionRepository, RuleEngine
from ..logger import log_execution_time, logger


@dataclass
class AutofillConfig:
    chunk_size: int
    parallel_processing: bool
    max_workers: int
    allow_missing_columns: bool
    seqn_column: str
    questions_dir: Path
    data_dir: Path
    output_dir: Path

    def __post_init__(self):
        if self.chunk_size < 1:
            raise ValueError("chunk_size must bes positive")
        if self.parallel_processing and not self.max_workers:
            raise ValueError("max_workers required when parallel_processing is enabled")
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
        self.data_df: Optional[pd.DataFrame] = None
        self.autofill_counts: Dict[str, int] = {}

    def process_file(self, input_file: Path, output_file: Path) -> None:
        self._load_data(input_file)
        self._validate_data()
        self._process_rules()
        self._save_results(output_file)

    def _load_data(self, input_file: Path) -> None:
        try:
            self.data_df = self.data_reader.read_csv(input_file)
            logger.info(f"Loaded {len(self.data_df)} records from {input_file}")
        except Exception as e:
            raise AutofillException(
                AutofillErrorCode.FILE_ERROR,
                f"Failed to load data: {str(e)}",
                source_file=input_file,
            )

    def _validate_data(self) -> None:
        if self.data_df is None:
            raise AutofillException(
                AutofillErrorCode.VALIDATION_ERROR, "No data loaded"
            )

        if (
            self.config.seqn_column not in self.data_df.columns
            and not self.config.allow_missing_columns
        ):
            raise AutofillException(
                AutofillErrorCode.VALIDATION_ERROR,
                f"Missing required column: {self.config.seqn_column}",
            )

    def _process_rules(self) -> None:
        if self.config.parallel_processing:
            self._parallel_process_chunks()
        else:
            self._sequential_process_chunks()

    def _parallel_process_chunks(self) -> None:
        """Process chunks in parallel using ThreadPoolExecutor."""
        if self.data_df is None:
            raise AutofillException(
                AutofillErrorCode.PROCESSING_ERROR, "No data loaded for processing"
            )

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            for start_idx in range(0, len(self.data_df), self.config.chunk_size):
                chunk = self.data_df.iloc[
                    start_idx : start_idx + self.config.chunk_size
                ]
                futures.append(executor.submit(self._process_chunk, chunk, start_idx))

            # Wait for all futures to complete and handle any exceptions
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    raise AutofillException(
                        AutofillErrorCode.PROCESSING_ERROR,
                        f"Parallel processing failed: {str(e)}",
                    )

    def _sequential_process_chunks(self) -> None:
        """Process chunks sequentially."""
        for start_idx in range(0, len(self.data_df), self.config.chunk_size):
            chunk = self.data_df.iloc[start_idx : start_idx + self.config.chunk_size]
            self._process_chunk(chunk, start_idx)

    def _process_chunk(self, chunk: pd.DataFrame, start_idx: int) -> None:
        try:
            logger.debug(
                f"Starting chunk processing:\n"
                f"Index range: {start_idx} to {start_idx + len(chunk)}\n"
                f"Chunk shape: {chunk.shape}\n"
                f"Memory usage: {chunk.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
            )

            questions_data = self.question_repository.load_questions(
                self.config.questions_dir
            )
            logger.debug(f"Loaded {len(questions_data)} questions for processing")

            processed_chunk = self.rule_engine.process_chunk(chunk, questions_data)
            self.data_df.iloc[start_idx : start_idx + len(chunk)] = processed_chunk

            logger.debug(f"Successfully processed chunk at index {start_idx}")

        except Exception as e:
            error_msg = (
                f"Error processing chunk at index {start_idx}:\n"
                f"Error type: {type(e).__name__}\n"
                f"Error details: {str(e)}\n"
                f"Chunk columns: {list(chunk.columns)}\n"
                f"Data types: {chunk.dtypes.to_dict()}"
            )
            logger.error(error_msg)
            raise AutofillException(
                AutofillErrorCode.PROCESSING_ERROR,
                f"Error processing chunk at index {start_idx}: {str(e)}",
            )

    def _save_results(self, output_file: Path) -> None:
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            self.data_df.to_csv(output_file, index=False)
            logger.info(f"Successfully saved results to {output_file}")
        except Exception as e:
            raise AutofillException(
                AutofillErrorCode.FILE_ERROR,
                f"Failed to save results: {str(e)}",
                source_file=output_file,
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.data_df is not None:
            del self.data_df
        return False

    @property
    def records_processed(self) -> int:
        return len(self.data_df) if self.data_df is not None else 0
