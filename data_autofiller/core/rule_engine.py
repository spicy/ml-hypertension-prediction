from decimal import Decimal
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..logger import logger
from .exceptions import AutofillErrorCode, AutofillException
from .interfaces import RuleEngine
from .tokens import TokenProcessor


class DefaultRuleEngine(RuleEngine):
    """
    Rule Engine for automated data filling based on predefined rules.

    The engine processes data in the following sequence:
    1. Processes data chunk by chunk
    2. For each chunk, identifies columns that have associated rules
    3. For each relevant column:
        - Finds rows with non-null values (trigger values)
        - Applies matching rules to auto-fill other columns in the same row
        - Only fills target cells that are currently empty

    Example rule structure:
    {
        "age": {  # source column
            "mappings": {
                "65-999": {  # if age is 65 or higher
                    "skip": {
                        "auto_fill": {
                            "other_question_1": "Yes",
                            "other_question_2": "No"
                        }
                    }
                }
            }
        }
    }
    """

    def __init__(self):
        self.token_processor = TokenProcessor()
        self.current_answer: Optional[Union[str, int, float]] = None
        self._cache = {}
        self._metrics = {
            "total_rules_processed": 0,
            "successful_matches": 0,
            "cache_hits": 0,
        }

    def process_rules(
        self, question_id: str, answer: Any, questions_data: Dict
    ) -> Dict[str, Dict]:
        """Process rules for a given question and answer."""
        if not question_id or pd.isna(answer):
            return {}

        try:
            question_mappings = questions_data.get(question_id, {}).get("mappings", {})
            if not question_mappings:
                logger.debug(f"No mappings found for question {question_id}")
                return {}

            self.current_answer = answer
            cache_key = (question_id, str(answer))

            if cache_key in self._cache:
                self._metrics["cache_hits"] += 1
                logger.debug(f"Cache hit for question {question_id}, answer {answer}")
                return self._cache[cache_key]

            self._metrics["total_rules_processed"] += 1

            try:
                result = self._process_numeric_answer(answer, question_mappings)
            except (ValueError, TypeError):
                result = self._process_string_answer(answer, question_mappings)

            if result:
                # Parse any string representations of dictionaries
                processed_result = {}
                for key, value in result.items():
                    if isinstance(value, str) and value.startswith("{"):
                        try:
                            import ast

                            parsed_value = ast.literal_eval(value)
                            processed_result[key] = {
                                "value": str(parsed_value.get("value", "")),
                                "overwrite_existing": parsed_value.get(
                                    "overwrite_existing", False
                                ),
                            }
                        except (ValueError, SyntaxError) as e:
                            logger.warning(f"Failed to parse value for {key}: {str(e)}")
                            continue
                    else:
                        processed_result[key] = value

                self._metrics["successful_matches"] += 1
                logger.info(
                    f"Rule match found for {question_id}: {answer} -> {processed_result}"
                )
                self._cache[cache_key] = processed_result
                return processed_result

            return {}

        except Exception as e:
            logger.error(f"Error processing rules for {question_id}: {str(e)}")
            return {}

    def process_chunk(self, df: pd.DataFrame, questions_data: Dict) -> pd.DataFrame:
        """Process a chunk of data applying rules and formulas."""
        try:
            processed_df = df.copy()
            relevant_columns = set(questions_data.keys()) & set(processed_df.columns)

            logger.debug(f"Processing chunk with columns: {list(relevant_columns)}")
            logger.debug(f"Questions data keys: {list(questions_data.keys())}")
            logger.debug(f"Data types in chunk: {processed_df.dtypes.to_dict()}")

            # First pass: Process regular autofill rules
            for question_id in questions_data:
                if "formula" not in questions_data[question_id]:
                    logger.debug(f"Processing question_id: {question_id}")
                    logger.debug(
                        f"Sample data for {question_id}: {processed_df[question_id].head()}"
                    )
                    logger.debug(
                        f"Question mappings: {questions_data[question_id].get('mappings', {})}"
                    )
                    processed_df = self._apply_rules_to_column(
                        processed_df, question_id, questions_data
                    )

            # Second pass: Process formula-based fields in dependency order
            formula_fields = {
                qid: data for qid, data in questions_data.items() if "formula" in data
            }

            logger.info(f"Processing {len(formula_fields)} formula-based fields")

            # Process each row
            for idx in processed_df.index:
                row_data = processed_df.loc[idx].to_dict()
                row_data["index"] = idx
                self.token_processor.set_row_data(row_data)

                logger.debug(f"Processing formulas for row {idx}")

                # Process formulas in dependency order
                remaining_fields = formula_fields.copy()
                processed_fields = set()
                progress_made = False

                while remaining_fields:
                    for field_id, field_data in list(remaining_fields.items()):
                        try:
                            formula = field_data["formula"]
                            logger.debug(
                                f"Processing formula for {field_id}: {formula}"
                            )

                            dependencies = self.token_processor._extract_dependencies(
                                formula
                            )
                            logger.debug(f"Dependencies for {field_id}: {dependencies}")

                            # Check if all dependencies are available
                            deps_available = all(
                                dep in processed_fields or dep in processed_df.columns
                                for dep in dependencies
                            )

                            if deps_available:
                                logger.debug(
                                    f"All dependencies available for {field_id}"
                                )
                                result = self.token_processor.process_formula(formula)

                                if result is not None:
                                    processed_df.at[idx, field_id] = result
                                    self.token_processor.set_processed_value(
                                        field_id, result
                                    )
                                    processed_fields.add(field_id)
                                    del remaining_fields[field_id]
                                    progress_made = True
                                    logger.debug(
                                        f"Successfully calculated {field_id} = {result}"
                                    )
                                else:
                                    logger.debug(
                                        f"Formula evaluation failed for {field_id}"
                                    )
                                    del remaining_fields[field_id]
                            else:
                                logger.debug(
                                    f"Dependencies not yet available for {field_id}"
                                )

                        except Exception as e:
                            logger.error(
                                f"Error processing formula for {field_id}: {str(e)}"
                            )
                            del remaining_fields[field_id]

                    # Check if any progress was made in this iteration, prevent infinite loop
                    if not progress_made and remaining_fields:
                        logger.warning(
                            "No more formulas can be processed, breaking loop"
                        )
                        break
                    progress_made = False

            return processed_df

        except Exception as e:
            logger.error(
                "Chunk processing error details:\n"
                f"Error type: {type(e).__name__}\n"
                f"Error message: {str(e)}\n"
                f"Data types: {df.dtypes.to_dict()}\n"
                f"Sample data: {df.head(1).to_dict()}\n"
                f"Stack trace:",
                exc_info=True,
            )
            raise AutofillException(
                AutofillErrorCode.PROCESSING_ERROR,
                f"Error processing chunk: {type(e).__name__} - {str(e)}",
            )

    def _apply_rules_to_column(
        self, df: pd.DataFrame, question_id: str, questions_data: Dict
    ) -> pd.DataFrame:
        """Apply autofill rules to a specific column in the DataFrame."""
        try:
            processed_df = df.copy()
            question_data = questions_data.get(question_id, {})

            # Check if this is a formula-based question
            if "formula" in question_data:
                formula = question_data["formula"]
                for idx in processed_df.index:
                    # Get current row data
                    row_data = processed_df.loc[idx].to_dict()
                    self.token_processor.set_row_data(row_data)

                    try:
                        result = self.token_processor.process_formula(formula)
                        if result is not None:
                            processed_df.at[idx, question_id] = result
                    except Exception as e:
                        logger.warning(
                            f"Failed to process formula for {question_id}: {str(e)}"
                        )
                        continue

            # Continue with regular rule processing...
            mask = ~pd.isna(processed_df[question_id])

            logger.info(f"Processing rules for column '{question_id}':")
            logger.info(f"- Total rows: {len(processed_df)}")
            logger.info(f"- Rows with values to process: {mask.sum()}")

            if not mask.any():
                logger.info(f"- Skipping '{question_id}': no values to process")
                return processed_df

            answers = processed_df.loc[mask, question_id]
            autofill_counts = {
                "attempted": 0,
                "successful": 0,
                "skipped_existing": 0,
                "no_matching_rules": 0,
                "missing_columns": 0,
                "conversion_failed": 0,
            }

            # Log unique values being processed
            unique_values = answers.value_counts()
            logger.debug(
                f"Processing unique values in {question_id}: {dict(unique_values)}"
            )

            for idx, answer in answers.items():
                autofill_values = self.process_rules(
                    question_id, answer, questions_data
                )
                autofill_counts["attempted"] += 1

                if not autofill_values:
                    autofill_counts["no_matching_rules"] += 1
                    logger.debug(f"No matching rules found for {question_id}={answer}")
                    continue

                for target_col, fill_config in autofill_values.items():
                    if target_col not in processed_df.columns:
                        autofill_counts["missing_columns"] += 1
                        logger.debug(
                            f"Target column '{target_col}' not found in DataFrame"
                        )
                        continue

                    current_value = processed_df.at[idx, target_col]
                    should_fill = pd.isna(current_value) or fill_config.get(
                        "overwrite_existing", False
                    )

                    if should_fill:
                        value = fill_config["value"]
                        col_dtype = str(processed_df[target_col].dtype)

                        try:
                            # Handle empty strings for numeric columns
                            if value == "":
                                if "float" in col_dtype:
                                    processed_df.loc[idx, target_col] = np.nan
                                elif "int" in col_dtype:
                                    processed_df.loc[idx, target_col] = pd.NA
                                autofill_counts["successful"] += 1
                                logger.debug(
                                    f"-> Filled '{target_col}' with NA/NaN (empty string converted)"
                                )
                            else:
                                value_series = pd.Series([value], dtype=col_dtype)
                                processed_df.loc[idx, target_col] = value_series[0]
                                autofill_counts["successful"] += 1
                                logger.debug(f"-> Filled '{target_col}' with '{value}'")
                        except (ValueError, TypeError) as e:
                            autofill_counts["conversion_failed"] += 1
                            logger.warning(
                                f"Failed to set value '{value}' for column '{target_col}' "
                                f"with dtype {col_dtype}: {str(e)}"
                            )
                            continue
                    else:
                        autofill_counts["skipped_existing"] += 1
                        logger.debug(
                            f"-> Skipped '{target_col}': already contains '{current_value}'"
                        )

            logger.info(
                f"\nAutofill summary for '{question_id}':"
                f"\n- Rules attempted: {autofill_counts['attempted']}"
                f"\n- No matching rules: {autofill_counts['no_matching_rules']}"
                f"\n- Missing columns: {autofill_counts['missing_columns']}"
                f"\n- Conversion failures: {autofill_counts['conversion_failed']}"
                f"\n- Successful fills: {autofill_counts['successful']}"
                f"\n- Skipped (existing values): {autofill_counts['skipped_existing']}"
            )

            return processed_df

        except Exception as e:
            logger.error(
                f"Error applying rules to column {question_id}: {str(e)}", exc_info=True
            )
            raise AutofillException(
                AutofillErrorCode.PROCESSING_ERROR,
                f"Error processing column {question_id}: {str(e)}",
            )

    def _process_answer(self, answer: Any, mappings: Dict) -> Dict[str, str]:
        """Process an answer using appropriate handling based on type."""
        try:
            if isinstance(answer, (int, float, np.number)):
                return self._process_numeric_answer(answer, mappings)
            return self._process_string_answer(str(answer), mappings)
        except Exception as e:
            logger.error(f"Error processing answer {answer}: {str(e)}")
            return {}

    def _process_numeric_answer(
        self, answer: Union[int, float, np.number], mappings: Dict
    ) -> Dict[str, str]:
        """Process numeric answers with range support."""
        try:
            # Preserve integer format
            if isinstance(answer, (int, np.integer)):
                answer_decimal = Decimal(str(int(answer)))
            elif (
                isinstance(answer, (float, np.floating)) and float(answer).is_integer()
            ):
                answer_decimal = Decimal(str(int(float(answer))))
            else:
                answer_decimal = Decimal(str(answer))

            for mapping_key, mapping_value in mappings.items():
                str_key = str(mapping_key)

                # Handle range values
                if "-" in str_key:
                    start_str, end_str = str_key.split("-")
                    try:
                        start = Decimal(start_str.strip())
                        end = Decimal(end_str.strip())
                        if start <= answer_decimal <= end:
                            result = self._get_autofill_values(mapping_value)
                            # Ensure integers stay as integers in the result
                            return {
                                k: (
                                    str(int(float(v)))
                                    if isinstance(v, (int, float))
                                    and float(v).is_integer()
                                    else str(v)
                                )
                                for k, v in result.items()
                            }
                    except (ValueError, ArithmeticError):
                        continue

                # Handle exact matches
                try:
                    if answer_decimal == Decimal(str_key):
                        result = self._get_autofill_values(mapping_value)
                        # Ensure integers stay as integers in the result
                        return {
                            k: (
                                str(int(float(v)))
                                if isinstance(v, (int, float)) and float(v).is_integer()
                                else str(v)
                            )
                            for k, v in result.items()
                        }
                except (ValueError, ArithmeticError):
                    continue

            return {}

        except Exception as e:
            logger.error(
                "Error processing numeric answer",
                extra={
                    "answer": answer,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return {}

    def _is_numeric_match(self, answer: Decimal, mapping_key: str) -> bool:
        """Check if numeric answer matches a mapping key."""
        try:
            if "-" in str(mapping_key):
                start, end = self._parse_range(mapping_key)
                return start <= answer <= end
            return answer == Decimal(str(mapping_key))
        except Exception:
            return False

    @staticmethod
    def _parse_range(range_str: str) -> Tuple[Decimal, Decimal]:
        """Parse a range string into start and end values."""
        start, end = map(str.strip, range_str.split("-"))
        return Decimal(start), Decimal(end)

    def _process_string_answer(self, answer: str, mappings: Dict) -> Dict[str, str]:
        """Process string-based answers with case-insensitive matching."""
        answer_normalized = answer.strip().lower()

        for mapping_key, mapping_value in mappings.items():
            if str(mapping_key).strip().lower() == answer_normalized:
                return self._get_autofill_values(mapping_value)
        return {}

    def _get_autofill_values(self, mapping_value: Dict) -> Dict[str, Dict]:
        """Process mapping values and handle special tokens."""
        processed_values = {}

        # Extract auto_fill values from the nested structure
        if isinstance(mapping_value, dict):
            if "skip" in mapping_value and "auto_fill" in mapping_value["skip"]:
                autofill_dict = mapping_value["skip"]["auto_fill"]
            else:
                return {}  # Return empty if no auto_fill found

            for target_col, value_config in autofill_dict.items():
                try:
                    # Parse the value_config if it's a string representation of a dict
                    if isinstance(value_config, str) and value_config.startswith("{"):
                        import ast

                        try:
                            value_config = ast.literal_eval(value_config)
                        except (ValueError, SyntaxError) as e:
                            logger.warning(
                                f"Failed to parse value_config for {target_col}: {str(e)}"
                            )
                            continue

                    # Handle both old and new format
                    if isinstance(value_config, (str, int, float)):
                        processed_values[target_col] = {
                            "value": str(value_config),
                            "overwrite_existing": False,
                        }
                    else:
                        value = str(value_config.get("value", ""))
                        overwrite_existing = value_config.get(
                            "overwrite_existing", False
                        )

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
                                processed_values[target_col] = {
                                    "value": str(processed_value),
                                    "overwrite_existing": overwrite_existing,
                                }
                        else:
                            processed_values[target_col] = {
                                "value": value,
                                "overwrite_existing": overwrite_existing,
                            }
                except Exception as e:
                    logger.warning(f"Error processing value for {target_col}: {str(e)}")
                    continue

        return processed_values

    def get_metrics(self) -> Dict[str, int]:
        """Return current processing metrics."""
        return {
            **self._metrics,
            "cache_size": len(self._cache),
            "cache_hit_ratio": (
                self._metrics["cache_hits"] / self._metrics["total_rules_processed"]
                if self._metrics["total_rules_processed"] > 0
                else 0
            ),
        }

    def clear_cache(self) -> None:
        """Clear the internal caches and reset metrics."""
        self._cache.clear()
        self._process_numeric_answer.cache_clear()
        self._metrics = {k: 0 for k in self._metrics}
