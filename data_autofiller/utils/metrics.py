import time
from dataclasses import dataclass
from typing import Dict


@dataclass
class PerformanceMetrics:
    start_time: float
    end_time: float = 0.0
    processed_records: int = 0
    processing_errors: int = 0
    memory_usage: Dict[str, float] = None

    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time

    def record_memory_usage(self, df_size: float):
        self.memory_usage = {"dataframe_size_mb": df_size / (1024 * 1024)}
