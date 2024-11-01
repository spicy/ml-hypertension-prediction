import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..core.exceptions import AutofillErrorCode, AutofillException
from ..core.interfaces import DataReader, QuestionRepository, RuleEngine
from ..logger import logger


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

    def process_file(self, input_file: Path, output_file: Path) -> None:
        """Process file in chunks to minimize memory usage."""
        questions_data = self.question_repository.load_questions(
            self.config.questions_dir
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Process the file in chunks
        first_chunk = True
        chunk_count = 0
        for chunk in pd.read_csv(input_file, chunksize=self.config.chunk_size):
            if chunk.empty:
                logger.warning(f"Empty chunk encountered in {input_file}")
                continue

            chunk_count += 1
            if first_chunk:
                self._validate_chunk(chunk)
                self._process_and_save_chunk(
                    chunk, output_file, questions_data, write_header=True
                )
                first_chunk = False
            else:
                self._process_and_save_chunk(
                    chunk, output_file, questions_data, write_header=False
                )

            self.records_processed += len(chunk)

        if chunk_count == 0:
            raise AutofillException(
                AutofillErrorCode.PROCESSING_ERROR,
                f"No valid chunks found in file: {input_file}",
            )

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

    def _process_and_save_chunk(
        self,
        chunk: pd.DataFrame,
        output_file: Path,
        questions_data: Dict,
        write_header: bool,
    ) -> None:
        """Process a chunk and append it to the output file."""
        try:
            start_time = time.time()
            processed_chunk = self.rule_engine.process_chunk(chunk, questions_data)

            # Record memory usage and processing time
            memory_mb = chunk.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.debug(
                f"Chunk processed: {len(chunk)} records, "
                f"Memory usage: {memory_mb:.2f} MB, "
                f"Processing time: {time.time() - start_time:.2f}s"
            )

            processed_chunk.to_csv(
                output_file,
                mode="w" if write_header else "a",
                header=write_header,
                index=False,
            )
        except Exception as e:
            raise AutofillException(
                AutofillErrorCode.PROCESSING_ERROR, f"Error processing chunk: {str(e)}"
            )
