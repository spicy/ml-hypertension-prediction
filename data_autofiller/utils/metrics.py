import time
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd
import psutil


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


@dataclass
class ProcessingMetrics:
    chunk_processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    rules_processed: int = 0
    cache_hits: int = 0
    peak_memory_usage: float = 0.0
    rule_processing_times: Dict[str, float] = field(default_factory=dict)
    cascade_iterations: int = 0
    vectorized_operations: int = 0
    parallel_operations: int = 0

    def update_metrics(self, chunk: pd.DataFrame, start_time: float):
        self.chunk_processing_time = time.time() - start_time
        self.memory_usage_mb = chunk.memory_usage(deep=True).sum() / (1024 * 1024)
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        self.peak_memory_usage = max(self.peak_memory_usage, current_memory)
