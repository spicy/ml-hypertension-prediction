from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ProcessingResult:
    success: bool
    records_processed: int
    errors: List[str]
    file_path: Optional[Path] = None
    processing_time: float = 0.0
    autofill_counts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.records_processed < 0:
            raise ValueError("records_processed cannot be negative")
        if self.processing_time < 0:
            raise ValueError("processing_time cannot be negative")


@dataclass
class BatchProcessingResult:
    total_files: int
    successful_files: int
    failed_files: int
    results: List[ProcessingResult]
    total_processing_time: float

    def dict(self) -> Dict:
        return {
            "total_files": self.total_files,
            "successful_files": self.successful_files,
            "failed_files": self.failed_files,
            "total_processing_time": self.total_processing_time,
            "results": [
                {
                    "success": r.success,
                    "records_processed": r.records_processed,
                    "errors": r.errors,
                    "file_path": str(r.file_path) if r.file_path else None,
                    "processing_time": r.processing_time,
                    "autofill_counts": r.autofill_counts,
                }
                for r in self.results
            ],
        }
