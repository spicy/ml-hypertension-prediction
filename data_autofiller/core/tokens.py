import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

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
        self._safe_operators = {
            "+",
            "-",
            "*",
            "/",
            "(",
            ")",
            ".",
            "=",
            "<",
            ">",
            "if",
            "else",
            "or",
            "and",
        }
        self._current_row_data: Dict[str, Any] = {}
        self._processed_values: Dict[str, Any] = {}
        self._processing_stack: List[str] = []  # Track formula processing stack

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
        # Only clear processed values if starting a new row
        if not hasattr(
            self, "_current_row_index"
        ) or self._current_row_index != row_data.get("index"):
            self._processed_values = {}
            self._current_row_index = row_data.get("index")

    def get_value(self, column: str) -> Any:
        """Get value from either processed values or original row data."""
        return self._processed_values.get(column, self._current_row_data.get(column))

    def set_processed_value(self, column: str, value: Any) -> None:
        """Store a processed value."""
        self._processed_values[column] = value

    def process_formula(self, formula: str) -> Optional[float]:
        """Process a formula with dependency handling."""
        try:
            logger.debug(f"Processing formula: {formula}")
            processed_formula = formula
            dependencies = self._extract_dependencies(formula)

            # Check for circular dependencies
            for dep in dependencies:
                if dep in self._processing_stack:
                    logger.error(
                        f"Circular dependency detected: {' -> '.join(self._processing_stack + [dep])}"
                    )
                    return None

            # Process dependencies first
            self._processing_stack.append(formula)
            for dep in dependencies:
                dep_value = self.get_value(dep)
                if pd.isna(dep_value):
                    logger.debug(f"Dependency {dep} has NA value, skipping formula")
                    self._processing_stack.pop()
                    return None

                processed_formula = processed_formula.replace(
                    f"##{dep}", str(float(dep_value))
                )

            self._processing_stack.pop()

            # For conditional expressions, use a different validation approach
            if "if" in formula or "else" in formula:
                # Validate only the numeric and operator parts
                parts = processed_formula.split()
                for part in parts:
                    if not (
                        part in {"if", "else"}
                        or self._is_numeric(part)
                        or all(c in self._safe_operators for c in part)
                    ):
                        raise ValueError(f"Invalid characters in formula part: {part}")
            else:
                # Regular mathematical formula validation
                if not all(
                    c.isspace() or self._is_numeric(c) or c in self._safe_operators
                    for c in processed_formula
                ):
                    raise ValueError(
                        f"Invalid characters in formula: {processed_formula}"
                    )

            # Create safe evaluation environment
            safe_dict = {
                "__builtins__": None,
                "True": True,
                "False": False,
            }

            result = eval(processed_formula, {"__builtins__": None}, safe_dict)
            logger.debug(f"Formula {formula} evaluated to {result}")
            return float(result)

        except Exception as e:
            logger.error(f"Error processing formula '{formula}': {str(e)}")
            return None

    def _extract_dependencies(self, formula: str) -> List[str]:
        """Extract column dependencies from a formula."""
        return [token[2:] for token in formula.split() if token.startswith("##")]

    def _is_numeric(self, value: str) -> bool:
        """Check if a string represents a valid number."""
        try:
            float(value)
            return True
        except ValueError:
            return False
