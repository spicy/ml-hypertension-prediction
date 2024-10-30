from enum import Enum
from typing import Callable, Optional


class AutofillToken(Enum):
    """Enumeration of special autofill tokens."""

    VALUE = "##VALUE"  # Copy the original answer value


class TokenProcessor:
    """Handles the processing of special autofill tokens."""

    @staticmethod
    def process_value(value: str) -> str:
        """Direct value copy."""
        return value

    @classmethod
    def get_processor(
        cls, token: AutofillToken | str
    ) -> Optional[Callable[[str], str]]:
        """Get the appropriate processor function for a token."""
        processors = {AutofillToken.VALUE: cls.process_value}
        if isinstance(token, str):
            token = next((t for t in AutofillToken if t.value == token), token)
        return processors.get(token)
