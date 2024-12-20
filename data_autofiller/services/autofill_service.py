import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from ..core.autofiller import AutofillConfig, DataAutofiller
from ..core.exceptions import AutofillException
from ..core.interfaces import DataReader, QuestionRepository, RuleEngine
from ..logger import log_execution_time, logger
from ..models.results import BatchProcessingResult, ProcessingResult


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

    @log_execution_time
    def process_files(
        self, input_files: List[Path], output_dir: Path
    ) -> BatchProcessingResult:
        start_time = time.time()
        results: List[ProcessingResult] = []

        if self.config.parallel_processing:
            results = self._process_files_parallel(input_files, output_dir)
        else:
            results = self._process_files_sequential(input_files, output_dir)

        successful = [r for r in results if r.success]

        return BatchProcessingResult(
            total_files=len(input_files),
            successful_files=len(successful),
            failed_files=len(results) - len(successful),
            results=results,
            total_processing_time=time.time() - start_time,
        )

    def _process_files_parallel(
        self, input_files: List[Path], output_dir: Path
    ) -> List[ProcessingResult]:
        results = []
        total_files = len(input_files)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._process_single_file, f, output_dir): f
                for f in input_files
            }

            for i, future in enumerate(as_completed(futures), 1):
                input_file = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(
                        f"Progress: {i}/{total_files} files ({input_file.name})"
                    )
                except Exception as e:
                    logger.error(f"Failed to process {input_file.name}: {str(e)}")
                    results.append(
                        ProcessingResult(
                            success=False,
                            records_processed=0,
                            errors=[str(e)],
                            file_path=input_file,
                        )
                    )

        return results

    def _process_files_sequential(
        self, input_files: List[Path], output_dir: Path
    ) -> List[ProcessingResult]:
        return [
            self._process_single_file(input_file, output_dir)
            for input_file in input_files
        ]

    def _process_single_file(
        self, input_file: Path, output_dir: Path
    ) -> ProcessingResult:
        start_time = time.time()
        try:
            output_file = self._generate_output_path(input_file, output_dir)
            self.autofiller.process_file(input_file, output_file)

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
