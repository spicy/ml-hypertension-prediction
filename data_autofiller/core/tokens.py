from enum import Enum
from typing import Callable, Dict, Optional


class AutofillToken(Enum):
    """Enumeration of special autofill tokens."""

    VALUE = "##VALUE"  # Copy the original answer value


class TokenProcessor:
    """Handles the processing of special autofill tokens."""

    def __init__(self):
        self._processors: Dict[AutofillToken, Callable] = {
            AutofillToken.VALUE: str,
        }

    def get_processor(self, token: str) -> Optional[Callable]:
        """Get the appropriate processor function for a token."""
        try:
            token_enum = AutofillToken(token)
            return self._processors.get(token_enum)
        except ValueError:
            return None
