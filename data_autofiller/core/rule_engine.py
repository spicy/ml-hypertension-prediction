import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..logger import logger
from ..utils.decorators import profile_performance
from .exceptions import AutofillErrorCode, AutofillException
from .interfaces import RuleEngine
from .tokens import TokenProcessor


class DefaultRuleEngine(RuleEngine):
    """
    Rule Engine for automated data filling based on predefined rules.

    The engine processes data in the following sequence:
    1. Processes data chunk by chunk with vectorized operations
    2. For each chunk, identifies columns that have associated rules
    3. For each relevant column:
        - Finds rows with non-null values (trigger values) using numpy masks
        - Applies matching rules to auto-fill other columns in the same row
        - Only fills target cells that are currently empty
        - Uses parallel processing for string operations
        - Uses vectorized operations for numeric comparisons

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

    def __init__(self, max_workers: int = 4, cache_size: int = 1000):
        self.token_processor = TokenProcessor()
        self.current_answer: Optional[Union[str, int, float]] = None
        self._cache = {}
        self._column_cache = {}
        self._metrics = {
            "total_rules_processed": 0,
            "successful_matches": 0,
            "cache_hits": 0,
            "vectorized_operations": 0,
        }
        self.max_workers = max_workers
        self._setup_caches(cache_size)

    def _setup_caches(self, cache_size: int) -> None:
        """Initialize LRU caches for different operations"""
        self._process_numeric_answer = lru_cache(maxsize=cache_size)(
            self._process_numeric_answer_impl
        )
        self._process_string_answer = lru_cache(maxsize=cache_size)(
            self._process_string_answer_impl
        )

    @profile_performance
    def process_chunk(self, chunk: pd.DataFrame, questions_data: Dict) -> pd.DataFrame:
        """Process a chunk of data with optimizations and detailed cascade tracking."""
        try:
            processed_chunk = self._optimize_dataframe_memory(chunk.copy())
            changes_made = True
            iteration = 0
            max_iterations = len(questions_data)
            self._metrics["cascade_iterations"] = 0

            # Track changes per iteration
            cascade_log = []

            while changes_made and iteration < max_iterations:
                changes_made = False
                iteration += 1
                self._metrics["cascade_iterations"] += 1

                iteration_changes = {
                    "iteration": iteration,
                    "changes": {},
                    "numeric_cols_processed": 0,
                    "string_cols_processed": 0,
                    "total_changes": 0,
                }

                # Track state before processing
                previous_state = processed_chunk.copy()

                # Process columns with optimizations
                relevant_columns = set(questions_data.keys()) & set(
                    processed_chunk.columns
                )
                numeric_cols, string_cols = self._categorize_columns(
                    processed_chunk, relevant_columns
                )

                if numeric_cols:
                    logger.debug(
                        f"Iteration {iteration}: Processing numeric columns {numeric_cols}"
                    )
                    processed_chunk = self._process_numeric_columns_vectorized(
                        processed_chunk, numeric_cols, questions_data
                    )
                    iteration_changes["numeric_cols_processed"] = len(numeric_cols)

                if string_cols:
                    logger.debug(
                        f"Iteration {iteration}: Processing string columns {string_cols}"
                    )
                    processed_chunk = self._process_string_columns_parallel(
                        processed_chunk, string_cols, questions_data
                    )
                    iteration_changes["string_cols_processed"] = len(string_cols)

                # Detect and log specific changes
                for col in relevant_columns:
                    if not processed_chunk[col].equals(previous_state[col]):
                        changes = (processed_chunk[col] != previous_state[col]).sum()
                        iteration_changes["changes"][col] = changes
                        iteration_changes["total_changes"] += changes
                        logger.debug(
                            f"Iteration {iteration}: Column {col} had {changes} values changed"
                        )

                changes_made = not processed_chunk.equals(previous_state)
                cascade_log.append(iteration_changes)

                if changes_made:
                    logger.info(
                        f"Iteration {iteration} completed with {iteration_changes['total_changes']} "
                        f"total changes across {len(iteration_changes['changes'])} columns"
                    )
                else:
                    logger.info(f"Iteration {iteration} completed with no changes")

            # Log final cascade summary
            self._log_cascade_summary(cascade_log)
            return processed_chunk

        except Exception as e:
            logger.error("Error in cascade processing", exc_info=True)
            raise AutofillException(
                AutofillErrorCode.PROCESSING_ERROR,
                f"Cascade processing error: {str(e)}",
            )

    @profile_performance
    def _process_numeric_columns_vectorized(
        self, df: pd.DataFrame, columns: List[str], questions_data: Dict
    ) -> pd.DataFrame:
        """Process numeric columns using vectorized operations"""
        try:
            for col in columns:
                mappings = questions_data.get(col, {}).get("mappings", {})
                if not mappings:
                    continue

                values = df[col].to_numpy()
                mask = ~pd.isna(values)

                if not mask.any():
                    continue

                logger.debug(
                    f"Processing numeric column {col} with vectorized operations"
                )

                for mapping_key, mapping_value in mappings.items():
                    match_mask = self._create_vectorized_match_mask(
                        values, mapping_key, mask
                    )

                    if match_mask.any():
                        self._apply_vectorized_rules(df, match_mask, mapping_value, col)
                        self._metrics["vectorized_operations"] += 1

            return df

        except Exception as e:
            logger.error(f"Error in vectorized processing: {str(e)}", exc_info=True)
            raise AutofillException(
                AutofillErrorCode.PROCESSING_ERROR,
                f"Vectorized processing error: {str(e)}",
            )

    def _process_string_columns_parallel(
        self, df: pd.DataFrame, columns: List[str], questions_data: Dict
    ) -> pd.DataFrame:
        """Process string columns using parallel execution"""
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for col in columns:
                    if col not in questions_data:
                        continue

                    logger.debug(
                        f"Submitting string column {col} for parallel processing"
                    )
                    futures.append(
                        executor.submit(
                            self._process_single_string_column,
                            df.copy(),
                            col,
                            questions_data[col],
                        )
                    )

                for future in futures:
                    try:
                        result = future.result()
                        if result is not None:
                            df.update(result)
                    except Exception as e:
                        logger.error(f"Error in parallel string processing: {str(e)}")

            return df

        except Exception as e:
            logger.error("Error in parallel processing", exc_info=True)
            raise AutofillException(
                AutofillErrorCode.PROCESSING_ERROR,
                f"Parallel processing error: {str(e)}",
            )

    @lru_cache(maxsize=100)
    def _create_vectorized_match_mask(
        self, values: np.ndarray, mapping_key: str, base_mask: np.ndarray
    ) -> np.ndarray:
        """Create a vectorized mask for matching values with caching"""
        try:
            if "-" in str(mapping_key):
                start, end = map(float, mapping_key.split("-"))
                return base_mask & (values >= start) & (values <= end)
            return base_mask & (values == float(mapping_key))
        except Exception as e:
            logger.warning(f"Error creating match mask: {str(e)}")
            return np.zeros_like(base_mask)

    def get_metrics(self) -> Dict[str, Any]:
        """Return enhanced processing metrics"""
        base_metrics = {
            **self._metrics,
            "cache_size": len(self._cache),
            "column_cache_size": len(self._column_cache),
            "cache_hit_ratio": (
                self._metrics["cache_hits"] / self._metrics["total_rules_processed"]
                if self._metrics["total_rules_processed"] > 0
                else 0
            ),
            "vectorized_operation_ratio": (
                self._metrics["vectorized_operations"]
                / self._metrics["total_rules_processed"]
                if self._metrics["total_rules_processed"] > 0
                else 0
            ),
        }
        return base_metrics

    def clear_cache(self) -> None:
        """Clear all caches and reset metrics"""
        self._cache.clear()
        self._column_cache.clear()
        self._process_numeric_answer.cache_clear()
        self._process_string_answer.cache_clear()
        self._create_vectorized_match_mask.cache_clear()
        self._metrics = {k: 0 for k in self._metrics}

    def _process_numeric_answer_impl(
        self, answer: Union[int, float, np.number], mapping_key: str
    ) -> Dict[str, str]:
        """Process numeric answers with optimized implementation"""
        try:
            if "-" in mapping_key:
                start, end = map(float, mapping_key.split("-"))
                if start <= float(answer) <= end:
                    return self._get_autofill_values(self._cache.get(mapping_key, {}))
            elif float(mapping_key) == float(answer):
                return self._get_autofill_values(self._cache.get(mapping_key, {}))
            return {}
        except (ValueError, TypeError) as e:
            logger.warning(f"Error processing numeric answer: {str(e)}")
            return {}

    def _process_string_answer_impl(
        self, answer: str, mapping_key: str
    ) -> Dict[str, str]:
        """Process string answers with optimized implementation"""
        try:
            if str(answer).lower() == str(mapping_key).lower():
                return self._get_autofill_values(self._cache.get(mapping_key, {}))
            return {}
        except Exception as e:
            logger.warning(f"Error processing string answer: {str(e)}")
            return {}

    def _process_single_string_column(
        self, df: pd.DataFrame, column: str, question_data: Dict
    ) -> Optional[pd.DataFrame]:
        """Process a single string column with optimized string operations"""
        try:
            mappings = question_data.get("mappings", {})
            if not mappings:
                return None

            mask = ~pd.isna(df[column])
            if not mask.any():
                return None

            self.current_answer = None
            for idx, answer in df.loc[mask, column].items():
                self.current_answer = answer
                for mapping_key, mapping_value in mappings.items():
                    autofill_values = self._process_string_answer(
                        str(answer), mapping_key
                    )
                    if autofill_values:
                        self._apply_rules_to_row(df, idx, autofill_values)

            return df
        except Exception as e:
            logger.error(f"Error processing string column {column}: {str(e)}")
            return None

    def _apply_vectorized_rules(
        self,
        df: pd.DataFrame,
        match_mask: np.ndarray,
        mapping_value: Dict,
        source_col: str,
    ) -> None:
        """Apply rules in a vectorized manner to matching rows"""
        try:
            autofill_values = self._get_autofill_values(mapping_value)
            for target_col, value in autofill_values.items():
                if target_col in df.columns:
                    fill_mask = match_mask & df[target_col].isna()
                    if fill_mask.any():
                        df.loc[fill_mask, target_col] = value
                        self._metrics["successful_matches"] += fill_mask.sum()
        except Exception as e:
            logger.error(f"Error applying vectorized rules: {str(e)}")

    def _categorize_columns(
        self, df: pd.DataFrame, columns: set
    ) -> Tuple[List[str], List[str]]:
        """Categorize columns into numeric and string types"""
        numeric_cols = []
        string_cols = []

        for col in columns:
            if np.issubdtype(df[col].dtype, np.number):
                numeric_cols.append(col)
            else:
                string_cols.append(col)

        return numeric_cols, string_cols

    @lru_cache(maxsize=1000)
    def _process_rule_cached(
        self, rule_key: str, value: Any, mapping_key: str
    ) -> Optional[Dict[str, str]]:
        """Cache individual rule processing results"""
        try:
            cache_key = f"{rule_key}:{value}:{mapping_key}"
            if cache_key in self._cache:
                self._metrics["cache_hits"] += 1
                return self._cache[cache_key]

            start_time = time.time()
            if isinstance(value, (int, float)):
                result = self._process_numeric_answer_impl(value, mapping_key)
            else:
                result = self._process_string_answer_impl(str(value), mapping_key)

            self._metrics["rule_processing_times"][rule_key] = time.time() - start_time
            self._cache[cache_key] = result
            return result
        except Exception as e:
            logger.warning(f"Error processing rule {rule_key}: {str(e)}")
            return None

    def _log_cascade_summary(self, cascade_log: List[Dict]) -> None:
        """Log detailed summary of the cascading operations."""
        total_changes = sum(log["total_changes"] for log in cascade_log)
        affected_columns = set()
        for log in cascade_log:
            affected_columns.update(log["changes"].keys())

        logger.info(
            "\nCascade Processing Summary:"
            f"\n- Total iterations: {len(cascade_log)}"
            f"\n- Total changes: {total_changes}"
            f"\n- Affected columns: {len(affected_columns)}"
            "\n\nIteration Details:"
        )

        for log in cascade_log:
            logger.info(
                f"\nIteration {log['iteration']}:"
                f"\n- Numeric columns processed: {log['numeric_cols_processed']}"
                f"\n- String columns processed: {log['string_cols_processed']}"
                f"\n- Total changes: {log['total_changes']}"
                f"\n- Changed columns: {list(log['changes'].keys())}"
            )
