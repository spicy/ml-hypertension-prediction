from typing import Dict, Any
from pathlib import Path
import json
import pandas as pd
import numpy as np
from logger import logger

def summarize_results(results: Dict[str, Any]) -> str:
    summary = []
    for analyzer, result in results.items():
        summary.append(f"{analyzer}: {len(result)} data points analyzed")
    return ", ".join(summary)

def save_results(results: Dict[str, Any], folder: str) -> None:
    def serialize_results(obj):
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