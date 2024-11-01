import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from config import data_config
from logger import logger


def summarize_results(results: dict[str, Any]) -> str:
    """
    Summarize the analysis results.
    """
    summary = []
    for analyzer, result in results.items():
        summary.append(f"{analyzer}: {len(result)} data points analyzed")
    return ", ".join(summary)


def save_results(results: dict[str, Any], folder: str) -> None:
    """
    Save analysis results to a JSON file.
    """

    def serialize_results(obj):
        """
        Serialize objects for JSON dumping.
        """
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.float64)):
            return int(obj) if isinstance(obj, np.int64) else float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    results_file = Path(folder) / data_config.RESULTS_FILENAME
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=serialize_results)
    logger.info(f"Results saved to {results_file}")
