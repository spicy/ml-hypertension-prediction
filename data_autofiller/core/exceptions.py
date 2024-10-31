from enum import Enum
from pathlib import Path
from typing import Optional


class AutofillErrorCode(Enum):
    INVALID_RULE = "INVALID_RULE"
    MISSING_DATA = "MISSING_DATA"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    FILE_ERROR = "FILE_ERROR"
    PROCESSING_ERROR = "PROCESSING_ERROR"


class AutofillException(Exception):
    """Base exception for all autofill-related errors."""

    def __init__(
        self,
        code: AutofillErrorCode,
        message: str,
        source_file: Optional[Path] = None,
        question_id: Optional[str] = None,
    ) -> None:
        self.code = code
        self.source_file = source_file
        self.question_id = question_id
        self.message = message
        super().__init__(self.formatted_message)

    @property
    def formatted_message(self) -> str:
        parts = [f"[{self.code.value}] {self.message}"]
        if self.source_file:
            parts.append(f"File: {self.source_file}")
        if self.question_id:
            parts.append(f"Question ID: {self.question_id}")
        return " | ".join(parts)


class InvalidRuleError(AutofillException):
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(AutofillErrorCode.INVALID_RULE, message, **kwargs)


class DataValidationError(AutofillException):
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(AutofillErrorCode.VALIDATION_ERROR, message, **kwargs)
