import time
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class ChunkMetrics:
    chunk_size: int
    memory_usage_mb: float
    processing_time: float


@dataclass
class PerformanceMetrics:
    start_time: float
    end_time: float = 0.0
    processed_records: int = 0
    processing_errors: int = 0
    chunk_metrics: List[ChunkMetrics] = field(default_factory=list)

    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time

    def add_chunk_metrics(self, chunk: pd.DataFrame, processing_time: float):
        memory_usage = chunk.memory_usage(deep=True).sum() / (1024 * 1024)
        self.chunk_metrics.append(
            ChunkMetrics(
                chunk_size=len(chunk),
                memory_usage_mb=memory_usage,
                processing_time=processing_time,
            )
        )
