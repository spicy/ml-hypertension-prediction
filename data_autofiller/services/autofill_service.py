import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import List

import pandas as pd

from ..core.autofiller import AutofillConfig, DataAutofiller
from ..core.exceptions import AutofillException
from ..core.include_filter import IncludeFilter
from ..core.interfaces import DataReader, QuestionRepository, RuleEngine
from ..logger import log_execution_time, logger
from ..models.results import BatchProcessingResult, ProcessingResult
from ..utils.metrics import PerformanceMetrics


class AutofillService:
    def __init__(
        self,
        data_reader: DataReader,
        question_repository: QuestionRepository,
        rule_engine: RuleEngine,
        config: AutofillConfig,
    ):
        self.autofiller = DataAutofiller(
            data_reader, question_repository, rule_engine, config
        )
        self.config = config
        self.metrics = PerformanceMetrics(start_time=time.time())

    @log_execution_time
    def process_files(
        self, input_files: List[Path], output_dir: Path
    ) -> BatchProcessingResult:
        start_time = time.time()
        results = self._process_files_parallel(input_files, output_dir)

        successful = [r for r in results if r.success]

        self.metrics.end_time = time.time()
        return BatchProcessingResult(
            total_files=len(input_files),
            successful_files=len(successful),
            failed_files=len(results) - len(successful),
            results=results,
            total_processing_time=self.metrics.total_time,
        )

    def _process_files_parallel(
        self, input_files: List[Path], output_dir: Path
    ) -> List[ProcessingResult]:
        max_workers = min(len(input_files), self.config.max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            process_func = partial(self._process_single_file, output_dir=output_dir)
            return list(executor.map(process_func, input_files))

    def _process_single_file(
        self, input_file: Path, output_dir: Path
    ) -> ProcessingResult:
        start_time = time.time()
        try:
            # Generate temporary file for unfiltered results
            temp_output = output_dir / f"temp_{input_file.name}"
            final_output = self._generate_output_path(input_file, output_dir)

            # Process the file with autofill rules
            self.autofiller.records_processed = 0
            self.autofiller.process_file(input_file, temp_output)

            # Load the complete processed file
            questions_data = self.autofiller.question_repository.load_questions(
                self.config.questions_dir
            )
            include_filter = IncludeFilter(questions_data)

            # Process the complete file with the include filter
            chunk_size = self.config.chunk_size
            for chunk_num, chunk in enumerate(
                pd.read_csv(temp_output, chunksize=chunk_size)
            ):
                filtered_chunk = include_filter.apply(chunk)
                # Write to final output file
                filtered_chunk.to_csv(
                    final_output,
                    mode="w" if chunk_num == 0 else "a",
                    header=chunk_num == 0,
                    index=False,
                )

            # Clean up temporary file
            temp_output.unlink()

            return ProcessingResult(
                success=True,
                records_processed=self.autofiller.records_processed,
                errors=[],
                file_path=input_file,
                processing_time=time.time() - start_time,
                autofill_counts=self.autofiller.autofill_counts,
            )
        except AutofillException as e:
            logger.error(str(e))
            return ProcessingResult(
                success=False,
                records_processed=0,
                errors=[str(e)],
                file_path=input_file,
                processing_time=time.time() - start_time,
            )
        except Exception as e:
            logger.exception(f"Unexpected error processing {input_file.name}")
            return ProcessingResult(
                success=False,
                records_processed=0,
                errors=[f"Unexpected error: {str(e)}"],
                file_path=input_file,
                processing_time=time.time() - start_time,
            )

    def _generate_output_path(self, input_file: Path, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        year_suffix = input_file.stem.split("_")[1]
        return output_dir / f"autofilled_data_{year_suffix}.csv"
