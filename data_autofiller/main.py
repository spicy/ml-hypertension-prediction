import sys
from os.path import abspath, dirname
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(dirname(dirname(abspath(__file__))))
import json
from datetime import datetime

from data_autofiller.config import Config
from data_autofiller.core.autofiller import AutofillConfig
from data_autofiller.core.rule_engine import DefaultRuleEngine
from data_autofiller.infrastructure.data_reader import FileDataReader
from data_autofiller.infrastructure.repositories import FileQuestionRepository
from data_autofiller.logger import logger
from data_autofiller.services.autofill_service import AutofillService
from data_autofiller.utils.file_utils import get_data_files


def save_processing_report(result, output_dir: Path) -> None:
    report = result.dict()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"processing_report_{timestamp}.json"

    with report_file.open("w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Processing report saved to {report_file}")


def main() -> int:
    try:
        app_config = Config()
        autofill_config = AutofillConfig(
            chunk_size=app_config.CHUNK_SIZE,
            parallel_processing=app_config.PARALLEL_PROCESSING,
            max_workers=app_config.MAX_WORKERS or 4,
            allow_missing_columns=app_config.ALLOW_MISSING_COLUMNS,
            seqn_column=app_config.SEQN_COLUMN,
            questions_dir=app_config.QUESTIONS_DIR,
            data_dir=app_config.DATA_DIR,
            output_dir=app_config.PROCESSED_DIR,
        )

        service = AutofillService(
            data_reader=FileDataReader(),
            question_repository=FileQuestionRepository(autofill_config.questions_dir),
            rule_engine=DefaultRuleEngine(),
            config=autofill_config,
        )

        input_files = get_data_files(
            autofill_config.data_dir, "FilteredCombinedData_*.csv"
        )

        result = service.process_files(input_files, autofill_config.output_dir)
        save_processing_report(result, autofill_config.output_dir)

        logger.info(
            f"Processing completed: {result.successful_files}/{result.total_files} files successful"
        )
        return 1 if result.failed_files > 0 else 0

    except Exception as e:
        logger.exception("Fatal error during data autofilling process")
        return 1


if __name__ == "__main__":
    sys.exit(main())
