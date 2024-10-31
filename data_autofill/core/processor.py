import logging
from typing import Dict

from .tokens import AutofillToken, TokenProcessor


class RuleProcessor:
    """Processes autofill rules and mappings."""

    def __init__(self):
        self.current_answer = None

    def process_autofill_rules(
        self, question_id: str, answer: str, questions_data: Dict
    ) -> Dict[str, str]:
        """Process autofill rules for a given question and answer."""
        if not (
            question_mappings := questions_data.get(question_id, {}).get("mappings", {})
        ):
            return {}

        self.current_answer = answer
        try:
            return self._process_numeric_answer(answer, question_mappings)
        except ValueError:
            return self._process_string_answer(answer, question_mappings)

    def _process_numeric_answer(self, answer: str, mappings: Dict) -> Dict[str, str]:
        """Process numeric answers and ranges."""
        answer_num = float(answer)

        for mapping_key, mapping_value in mappings.items():
            if "-" in str(mapping_key):
                range_start, range_end = map(float, mapping_key.split("-"))
                if range_start <= answer_num <= range_end:
                    return self._get_autofill_values(mapping_value)
            elif str(answer_num) == str(mapping_key):
                return self._get_autofill_values(mapping_value)
        return {}

    def _process_string_answer(self, answer: str, mappings: Dict) -> Dict[str, str]:
        """Process string answers."""
        if mapping_value := mappings.get(str(answer)):
            return self._get_autofill_values(mapping_value)
        return {}

    def _get_autofill_values(self, mapping_value: Dict) -> Dict[str, str]:
        """Extract autofill values from mapping, handling special tokens."""
        autofill_dict = mapping_value.get("skip", {}).get("auto_fill", {})
        processed_dict = {}

        for key, value in autofill_dict.items():
            if isinstance(value, str) and value == AutofillToken.VALUE.value:
                value = AutofillToken.VALUE

            if processor := TokenProcessor.get_processor(value):
                try:
                    processed_dict[key] = processor(self.current_answer)
                except (ValueError, TypeError) as e:
                    logging.warning(
                        f"Error processing token {value} for {key}: {str(e)}"
                    )
                    processed_dict[key] = value
            else:
                processed_dict[key] = value

        return processed_dict
