import json
from pathlib import Path
from typing import List

import pandas as pd

from ..config import config
from ..logger import logger
from .data_utils import convert_numeric_to_int64


def read_and_validate_file(file_path: Path) -> pd.DataFrame:
    """Read a CSV file and validate it has SEQN column."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    if config.SEQN_COLUMN not in df.columns:
        raise ValueError(f"File {file_path} missing '{config.SEQN_COLUMN}' column")

    return convert_numeric_to_int64(df)


def get_data_directories() -> List[Path]:
    """Get all year directories from the raw data directory."""
    logger.info(f"Looking for data in: {config.RAW_DATA_DIR}")
    if not config.RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {config.RAW_DATA_DIR}")

    def is_year_directory(path: Path) -> bool:
        """Check if directory name contains valid year format."""
        name = path.name
        # Match single year (e.g., "2017") or year range (e.g., "2017-2020")
        return (name.isdigit() and len(name) == 4) or (
            "-" in name
            and all(part.isdigit() and len(part) == 4 for part in name.split("-"))
        )

    dirs = sorted(
        path
        for path in config.RAW_DATA_DIR.iterdir()
        if path.is_dir() and is_year_directory(path)
    )
    logger.info(f"Found directories: {[str(d) for d in dirs]}")
    return dirs


def get_relevant_columns() -> List[str]:
    """Get a list of relevant columns from JSON files in the questions directory."""
    if not config.QUESTIONS_DIR.exists():
        raise FileNotFoundError(
            f"Questions directory not found: {config.QUESTIONS_DIR}"
        )

    relevant_columns = set([config.SEQN_COLUMN])

    for json_file in config.QUESTIONS_DIR.glob("*.json"):
        try:
            with json_file.open("r") as f:
                data = json.load(f)
                relevant_columns.update(data.keys())
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON file: {json_file}")
        except Exception as e:
            logger.error(f"Error processing file {json_file}: {str(e)}")

    if len(relevant_columns) == 1:
        logger.warning("No additional columns found in JSON files.")

    return list(relevant_columns)
