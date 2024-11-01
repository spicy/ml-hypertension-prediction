import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..logger import logger


def read_json_file(file_path: Path) -> Dict:
    """Read and parse a JSON file."""
    try:
        with file_path.open("r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON file: {file_path}")
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
    return {}


def read_csv_file(file_path: Path) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {str(e)}")
        raise


def get_data_files(directory: Path, pattern: str) -> List[Path]:
    """Get all files matching pattern from the specified directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(directory.glob(pattern))
    logger.info(f"Found {len(files)} files matching pattern '{pattern}' in {directory}")
    return files


def check_file_exists(file_path: Path) -> None:
    """Check if file exists and is a regular file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")


def check_file_readable(file_path: Path) -> None:
    """Verify file is readable and has content."""
    try:
        with file_path.open("r") as f:
            first_line = f.readline()
            if not first_line:
                raise ValueError(f"File is empty: {file_path}")
    except PermissionError:
        raise ValueError(f"Permission denied reading file: {file_path}")
    except UnicodeDecodeError:
        raise ValueError(f"File encoding error: {file_path}")
    except Exception as e:
        raise ValueError(f"Unable to read file {file_path}: {str(e)}")


def check_csv_format(file_path: Path) -> None:
    """Verify file has valid CSV format."""
    try:
        # Read just the header to validate CSV structure
        pd.read_csv(file_path, nrows=0)
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {file_path}")
    except pd.errors.ParserError:
        raise ValueError(f"Invalid CSV format in file: {file_path}")


def validate_input_file(file_path: Path) -> None:
    """Validate that input file exists and is a readable CSV file."""
    check_file_exists(file_path)
    check_file_readable(file_path)
    check_csv_format(file_path)
