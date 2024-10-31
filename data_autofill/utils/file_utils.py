import json
from pathlib import Path
from typing import Dict

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
