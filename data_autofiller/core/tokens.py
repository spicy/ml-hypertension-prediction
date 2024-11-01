import logging
from enum import Enum
from typing import Any, Callable, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class AutofillToken(Enum):
    """Enumeration of special autofill tokens."""

    VALUE = "##VALUE"  # Copy the original answer value
    FORMULA = "FORMULA"  # Indicates a formula calculation


class TokenProcessor:
    """Handles the processing of special autofill tokens."""

    def __init__(self):
        self._processors: Dict[AutofillToken, Callable] = {
            AutofillToken.VALUE: str,
            AutofillToken.FORMULA: self.process_formula,
        }
        self._safe_operators = {"+", "-", "*", "/", "(", ")", "."}
        self._current_row_data: Dict[str, Any] = {}

    def get_processor(self, token: str) -> Optional[Callable]:
        """Get the appropriate processor function for a token."""
        try:
            token_enum = AutofillToken(token)
            return self._processors.get(token_enum)
        except ValueError:
            return None

    def set_row_data(self, row_data: Dict[str, Any]) -> None:
        """Set the current row data for formula processing."""
        self._current_row_data = row_data

    def process_formula(self, formula: str) -> Optional[float]:
        """Process a formula safely with the current row data."""
        try:
            # Replace all tokens (##COLUMN_NAME) with their values
            processed_formula = formula
            for col_name, value in self._current_row_data.items():
                token = f"##{col_name}"
                if token in processed_formula:
                    if pd.isna(value):
                        return None
                    processed_formula = processed_formula.replace(
                        token, str(float(value))
                    )

            # Validate formula contains only safe characters
            if not all(
                c.isspace() or c.isdigit() or c in self._safe_operators
                for c in processed_formula
            ):
                raise ValueError(f"Invalid characters in formula: {formula}")

            # Evaluate the formula in a restricted environment
            result = eval(processed_formula, {"__builtins__": {}}, {})
            return float(result)

        except Exception as e:
            logger.error(f"Error processing formula '{formula}': {str(e)}")
            return None
