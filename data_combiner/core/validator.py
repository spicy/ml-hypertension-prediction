from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from ..config import config
from ..logger import logger


class ValidationErrorType(Enum):
    """Enumeration of possible validation error types."""

    MISSING_SEQN = auto()
    VALUE_MISMATCH = auto()
    NO_COMMON_COLUMNS = auto()


@dataclass
class ValidationError:
    """Data class to store validation error details."""

    source_file: Path
    seqn: int
    error_type: ValidationErrorType
    details: str = ""
    column: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None

    def __str__(self) -> str:
        """Format validation error message based on error type."""
        if self.error_type == ValidationErrorType.MISSING_SEQN:
            return (
                f"SEQN {self.seqn} from {self.source_file} not found in combined data"
            )
        elif self.error_type == ValidationErrorType.VALUE_MISMATCH:
            return (
                f"Mismatch for SEQN {self.seqn}, column {self.column} in {self.source_file}: "
                f"Expected {self.expected_value}, got {self.actual_value}"
            )
        return f"{self.error_type.name}: {self.details}"


class DataValidator:
    """Validates combined data against source files."""

    CHUNK_SIZE = 1000
    MAX_ERRORS_TO_DISPLAY = 10

    def __init__(self, combined_df: pd.DataFrame, source_files: List[Path]):
        """Initialize validator with combined data and source files."""
        self._combined_df = combined_df
        self._source_files = source_files
        self._validation_errors: List[ValidationError] = []
        self._seqn_index = None
        self._initialize_seqn_index()

    def _initialize_seqn_index(self) -> None:
        """Create an index for faster SEQN lookups."""
        self._seqn_index = pd.Index(self._combined_df[config.SEQN_COLUMN])
        logger.debug(f"Created SEQN index with {len(self._seqn_index)} entries")

    def _get_common_columns(self, source_df: pd.DataFrame) -> Set[str]:
        """Get common columns between source and combined DataFrames."""
        common_cols = set(source_df.columns) & set(self._combined_df.columns)
        logger.debug(f"Found {len(common_cols)} common columns: {sorted(common_cols)}")
        return common_cols

    def _validate_common_columns(
        self, source_file: Path, common_cols: Set[str]
    ) -> bool:
        """Validate that common columns exist between datasets."""
        if not common_cols:
            self._validation_errors.append(
                ValidationError(
                    source_file=source_file,
                    seqn=0,
                    error_type=ValidationErrorType.NO_COMMON_COLUMNS,
                    details="No common columns found",
                )
            )
            return False
        return True

    def _compare_row_values(
        self,
        source_row: pd.Series,
        combined_row: pd.Series,
        common_cols: Set[str],
        seqn: int,
        source_file: Path,
    ) -> None:
        """Compare values between source and combined rows."""
        for col in common_cols:
            source_val = source_row[col]
            combined_val = combined_row[col]

            if pd.isna(source_val) and pd.isna(combined_val):
                continue

            if source_val != combined_val:
                logger.warning(
                    f"WARNING: Value mismatch for SEQN {seqn}, column '{col}' in {source_file.name}\n"
                    f"  Expected: {source_val}\n"
                    f"  Got:      {combined_val}"
                )
                self._validation_errors.append(
                    ValidationError(
                        source_file=source_file,
                        seqn=seqn,
                        error_type=ValidationErrorType.VALUE_MISMATCH,
                        column=col,
                        expected_value=source_val,
                        actual_value=combined_val,
                    )
                )

    def _validate_chunk(
        self,
        chunk_df: pd.DataFrame,
        common_cols: Set[str],
        source_file: Path,
        start_idx: int,
        total_rows: int,
    ) -> None:
        """Validate a chunk of data from the source file."""
        simple_path = self._get_simplified_path(source_file)
        logger.debug(
            f"Validating rows {start_idx} to {start_idx + len(chunk_df)} "
            f"of {total_rows} in {simple_path}"
        )

        for _, source_row in chunk_df.iterrows():
            seqn = source_row[config.SEQN_COLUMN]

            if seqn not in self._seqn_index:
                logger.warning(
                    f"WARNING: SEQN {seqn} from {source_file} not found in combined dataset"
                )
                self._validation_errors.append(
                    ValidationError(
                        source_file=source_file,
                        seqn=seqn,
                        error_type=ValidationErrorType.MISSING_SEQN,
                    )
                )
                continue

            combined_row = self._combined_df.iloc[self._seqn_index.get_loc(seqn)]
            self._compare_row_values(
                source_row, combined_row, common_cols, seqn, source_file
            )

    def _validate_file_data(self, source_df: pd.DataFrame, source_file: Path) -> None:
        """Validate data from a single source file."""
        common_cols = self._get_common_columns(source_df)
        if not self._validate_common_columns(source_file, common_cols):
            return

        total_rows = len(source_df)
        for start_idx in range(0, total_rows, self.CHUNK_SIZE):
            chunk_df = source_df.iloc[
                start_idx : min(start_idx + self.CHUNK_SIZE, total_rows)
            ]
            self._validate_chunk(
                chunk_df, common_cols, source_file, start_idx, total_rows
            )

    def _get_simplified_path(self, source_file: Path) -> str:
        """Convert full path to 'year/filename' format."""
        # Gets the parent folder name (year) and filename
        return f"{source_file.parent.name}/{source_file.name}"

    def validate(self) -> None:
        """Perform validation and raise exception if errors are found."""
        for source_file in self._source_files:
            try:
                simple_path = self._get_simplified_path(source_file)
                logger.info(f"Starting validation of {simple_path}")
                source_df = pd.read_csv(source_file)
                logger.debug(
                    f"Loaded source file {simple_path} with {len(source_df)} rows"
                )
                self._validate_file_data(source_df, source_file)
                logger.info(f"Completed validation of {simple_path}")

            except Exception as e:
                logger.error(f"Error validating {source_file}: {str(e)}")
                raise

        if self._validation_errors:
            self._report_validation_errors()
            raise ValueError(f"Found {len(self._validation_errors)} validation errors")

        logger.info("Validation completed successfully")

    def _report_validation_errors(self) -> None:
        """Report validation errors in a structured format."""
        error_count = len(self._validation_errors)
        logger.error(f"Found {error_count} validation errors")

        # Group errors by type for better reporting
        errors_by_type: Dict[ValidationErrorType, List[ValidationError]] = {}
        for error in self._validation_errors:
            errors_by_type.setdefault(error.error_type, []).append(error)

        for error_type, errors in errors_by_type.items():
            logger.error(f"\n{error_type.name} errors ({len(errors)} total):")
            for error in errors[: self.MAX_ERRORS_TO_DISPLAY]:
                logger.error(f"  - {error}")
            if len(errors) > self.MAX_ERRORS_TO_DISPLAY:
                logger.error(
                    f"  ... and {len(errors) - self.MAX_ERRORS_TO_DISPLAY} more"
                )
