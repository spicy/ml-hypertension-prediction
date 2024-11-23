import logging
from typing import Dict

from .tokens import AutofillToken, TokenProcessor


class RuleProcessor:
    """Processes autofill rules and mappings."""

    def __init__(self):
        self.current_answer = None
        self.token_processor = TokenProcessor()

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

    def _get_autofill_values(self, mapping_value: Dict) -> Dict[str, Dict]:
        """Extract autofill values from mapping, handling special tokens."""
        autofill_dict = mapping_value.get("skip", {}).get("auto_fill", {})
        processed_dict = {}

        for key, value_config in autofill_dict.items():
            try:
                # Handle both old and new format
                if isinstance(value_config, (str, int, float)):
                    value = str(value_config)
                    overwrite_existing = False
                else:
                    value = str(value_config.get("value", ""))
                    overwrite_existing = value_config.get("overwrite_existing", False)

                if value.startswith("##"):
                    if "formula" in mapping_value:
                        processed_value = self.token_processor.process_formula(
                            mapping_value["formula"]
                        )
                    elif value == "##VALUE":
                        processed_value = self.current_answer
                    else:
                        processed_value = self.token_processor.process_token(
                            value, self.current_answer
                        )

                    if processed_value is not None:
                        processed_dict[key] = {
                            "value": str(processed_value),
                            "overwrite_existing": overwrite_existing,
                        }
                else:
                    processed_dict[key] = {
                        "value": value,
                        "overwrite_existing": overwrite_existing,
                    }
            except Exception as e:
                logging.warning(f"Error processing value for {key}: {str(e)}")
                continue

        return processed_dict
