from typing import Any
from pathlib import Path
import json
import pandas as pd
import numpy as np
from logger import logger

def summarize_results(results: dict[str, Any]) -> str:
    """
    Summarize the analysis results.

    Args:
        results (dict[str, Any]): A dictionary containing analysis results.

    Returns:
        str: A string summarizing the number of data points analyzed for each analyzer.
    """
    summary = []
    for analyzer, result in results.items():
        summary.append(f"{analyzer}: {len(result)} data points analyzed")
    return ", ".join(summary)

def save_results(results: dict[str, Any], folder: str) -> None:
    """
    Save analysis results to a JSON file.

    Args:
        results (dict[str, Any]): A dictionary containing analysis results.
        folder (str): The folder path where the results will be saved.

    Returns:
        None
    """
    def serialize_results(obj):
        """
        Serialize objects for JSON dumping.

        Args:
            obj: The object to be serialized.

        Returns:
            A JSON-serializable representation of the object.

        Raises:
            TypeError: If the object type is not JSON serializable.
        """
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.float64)):
            return int(obj) if isinstance(obj, np.int64) else float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    results_file = Path(folder) / "analysis_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=serialize_results)
    logger.info(f"Results saved to {results_file}")