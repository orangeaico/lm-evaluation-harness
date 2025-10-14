"""
Utilities for repo-eval tasks in lm-evaluation-harness.

This module provides data processing and metric calculation functions
for the DefinedIn, BelongsTo, CallsWhat, and CalledBy code understanding tasks.
Enhanced with fenced JSON parsing and Pydantic validation.
"""

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import datasets
from lm_eval.api import registry
from lm_eval.api.registry import register_metric, register_aggregation

# Ensure project src directory is on the path for parser imports
SRC_ROOT = Path(__file__).resolve().parents[5]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval.parsers.fenced_json_parser import (
    parse_defined_in,
    parse_belongs_to,
    parse_calls_what,
    parse_called_by,
    parse_loc_file,
    parse_loc_func,
    parse_loc_line,
    parse_code_edit,
    extract_fenced_json,
)


TASK_PARSERS = {
    'DefinedIn': parse_defined_in,
    'BelongsTo': parse_belongs_to,
    'CallsWhat': parse_calls_what,
    'CalledBy': parse_called_by,
    'LocFile': parse_loc_file,
    'LocFunc': parse_loc_func,
    'LocLine': parse_loc_line,
    'CodeEdit': parse_code_edit,
}


def process_defined_in_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Process DefinedIn task documents for lm-eval format.

    Args:
        dataset: Raw dataset from JSONL files

    Returns:
        Processed dataset with expected format
    """
    def _process_doc(doc):
        # Convert expected outputs to JSON string for comparison
        expected_outputs_json = json.dumps(doc["expected_outputs"], sort_keys=True)

        return {
            "prompt": doc["prompt"],
            "expected_outputs": doc["expected_outputs"],
            "expected_outputs_json": expected_outputs_json,
            "task_id": doc["task_id"],
            "repo_name": doc["repo_name"],
            "commit_hash": doc["commit_hash"],
            "difficulty_level": doc.get("difficulty_level", "unknown")
        }

    return dataset.map(_process_doc)


def process_belongs_to_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Process BelongsTo task documents for lm-eval format.

    Args:
        dataset: Raw dataset from JSONL files

    Returns:
        Processed dataset with expected format
    """
    def _process_doc(doc):
        # Convert expected outputs to JSON string for comparison
        expected_outputs_json = json.dumps(doc["expected_outputs"], sort_keys=True)

        return {
            "prompt": doc["prompt"],
            "expected_outputs": doc["expected_outputs"],
            "expected_outputs_json": expected_outputs_json,
            "task_id": doc["task_id"],
            "repo_name": doc["repo_name"],
            "commit_hash": doc["commit_hash"],
            "difficulty_level": doc.get("difficulty_level", "unknown")
        }

    return dataset.map(_process_doc)


def process_calls_what_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Process CallsWhat task documents for lm-eval format.

    Args:
        dataset: Raw dataset from JSONL files

    Returns:
        Processed dataset with expected format
    """
    def _process_doc(doc):
        # Convert expected outputs to JSON string for comparison
        expected_outputs_json = json.dumps(doc["expected_outputs"], sort_keys=True)

        return {
            "prompt": doc["prompt"],
            "expected_outputs": doc["expected_outputs"],
            "expected_outputs_json": expected_outputs_json,
            "task_id": doc["task_id"],
            "repo_name": doc["repo_name"],
            "commit_hash": doc["commit_hash"],
            "difficulty_level": doc.get("difficulty_level", "unknown")
        }

    return dataset.map(_process_doc)


def process_called_by_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Process CalledBy task documents for lm-eval format.

    Args:
        dataset: Raw dataset from JSONL files

    Returns:
        Processed dataset with expected format
    """
    def _process_doc(doc):
        # Convert expected outputs to JSON string for comparison
        expected_outputs_json = json.dumps(doc["expected_outputs"], sort_keys=True)

        return {
            "prompt": doc["prompt"],
            "expected_outputs": doc["expected_outputs"],
            "expected_outputs_json": expected_outputs_json,
            "task_id": doc["task_id"],
            "repo_name": doc["repo_name"],
            "commit_hash": doc["commit_hash"],
            "difficulty_level": doc.get("difficulty_level", "unknown")
        }

    return dataset.map(_process_doc)


def process_loc_file_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process LocFile documents for lm-eval format."""

    def _process_doc(doc):
        expected_outputs_json = json.dumps(doc["expected_outputs"], sort_keys=True)
        return {
            "prompt": doc["prompt"],
            "expected_outputs": doc["expected_outputs"],
            "expected_outputs_json": expected_outputs_json,
            "task_id": doc["task_id"],
            "repo_name": doc["repo_name"],
            "commit_hash": doc["commit_hash"],
            "difficulty_level": doc.get("difficulty_level", "unknown"),
        }

    return dataset.map(_process_doc)


def process_loc_func_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process LocFunc documents for lm-eval format."""

    def _process_doc(doc):
        expected_outputs_json = json.dumps(doc["expected_outputs"], sort_keys=True)
        return {
            "prompt": doc["prompt"],
            "expected_outputs": doc["expected_outputs"],
            "expected_outputs_json": expected_outputs_json,
            "task_id": doc["task_id"],
            "repo_name": doc["repo_name"],
            "commit_hash": doc["commit_hash"],
            "difficulty_level": doc.get("difficulty_level", "unknown"),
        }

    return dataset.map(_process_doc)


def process_loc_line_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process LocLine documents for lm-eval format."""

    def _process_doc(doc):
        expected_outputs_json = json.dumps(doc["expected_outputs"], sort_keys=True)
        return {
            "prompt": doc["prompt"],
            "expected_outputs": doc["expected_outputs"],
            "expected_outputs_json": expected_outputs_json,
            "task_id": doc["task_id"],
            "repo_name": doc["repo_name"],
            "commit_hash": doc["commit_hash"],
            "difficulty_level": doc.get("difficulty_level", "unknown"),
        }

    return dataset.map(_process_doc)


def process_code_edit_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process CodeEdit documents for lm-eval format."""

    def _process_doc(doc):
        expected_outputs_json = json.dumps(doc["expected_outputs"], sort_keys=True)
        return {
            "prompt": doc["prompt"],
            "expected_outputs": doc["expected_outputs"],
            "expected_outputs_json": expected_outputs_json,
            "task_id": doc["task_id"],
            "repo_name": doc["repo_name"],
            "commit_hash": doc["commit_hash"],
            "difficulty_level": doc.get("difficulty_level", "unknown"),
        }

    return dataset.map(_process_doc)


class FencedJSONFilter:
    """
    Filter class to extract and validate fenced JSON from model responses.

    This replaces the old ExtractJSONFilter with proper fenced JSON parsing
    and Pydantic validation for each task type.
    """

    def __init__(self, task_name: str):
        """
        Initialize filter for a specific task type.

        Args:
            task_name: One of the registered repo-eval task names (DefinedIn, BelongsTo,
                       CallsWhat, CalledBy, LocFile, LocFunc, LocLine)
        """
        self.task_name = task_name

    def apply(self, resps: List[str], docs: List[Dict[str, Any]]) -> List[str]:
        """
        Apply fenced JSON extraction and validation to model responses.

        Args:
            resps: List of model responses (may be nested lists)
            docs: List of documents (unused)

        Returns:
            List of validated JSON strings ready for metric calculation
        """
        parser = TASK_PARSERS.get(self.task_name)
        if not parser:
            raise ValueError(f"Unknown task name: {self.task_name}")

        processed = []
        for resp in resps:
            # Handle nested list responses from LM-eval
            if isinstance(resp, list):
                resp_str = str(resp[0]) if resp else ""
            else:
                resp_str = str(resp)

            # Parse and validate using task-specific parser
            parsed_dict = parser(resp_str)

            # Convert back to JSON string for LM-eval compatibility
            processed.append(json.dumps(parsed_dict))

        return processed


# Task-specific filter factory functions
def defined_in_json_filter():
    """Factory function for DefinedIn JSON extraction filter."""
    return FencedJSONFilter('DefinedIn')


def belongs_to_json_filter():
    """Factory function for BelongsTo JSON extraction filter."""
    return FencedJSONFilter('BelongsTo')


def calls_what_json_filter():
    """Factory function for CallsWhat JSON extraction filter."""
    return FencedJSONFilter('CallsWhat')


def called_by_json_filter():
    """Factory function for CalledBy JSON extraction filter."""
    return FencedJSONFilter('CalledBy')


def loc_file_json_filter():
    """Factory function for LocFile JSON extraction filter."""
    return FencedJSONFilter('LocFile')


def loc_func_json_filter():
    """Factory function for LocFunc JSON extraction filter."""
    return FencedJSONFilter('LocFunc')


def loc_line_json_filter():
    """Factory function for LocLine JSON extraction filter."""
    return FencedJSONFilter('LocLine')


# Legacy function for backward compatibility
def extract_json_response():
    """Legacy factory function - use task-specific filters instead."""
    return FencedJSONFilter('DefinedIn')  # Default fallback


def _parse_model_output(response: str, task_name: str) -> Dict[str, Any]:
    """
    Parse model output and extract JSON for a specific task.

    Args:
        response: Model response string
        task_name: Name of the task to determine the expected schema

    Returns:
        Parsed JSON dict or empty dict if parsing fails
    """
    parser = TASK_PARSERS.get(task_name)
    if not parser:
        raise ValueError(f"Unknown task name: {task_name}")

    try:
        return parser(response)
    except Exception:
        return {}


def defined_in_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Calculate exact match accuracy for DefinedIn task.

    Args:
        predictions: List of model predictions
        references: List of reference JSON strings

    Returns:
        Exact match accuracy (0.0 to 1.0)
    """
    correct = 0
    total = len(predictions)

    for pred, ref in zip(predictions, references):
        try:
            pred_json = _parse_model_output(pred, 'DefinedIn')
            ref_json = json.loads(ref)

            # Check if all required fields match exactly (filename and class)
            if (pred_json.get("filename") == ref_json.get("filename") and
                pred_json.get("class") == ref_json.get("class")):
                correct += 1

        except (json.JSONDecodeError, AttributeError):
            # Prediction couldn't be parsed as JSON
            pass

    return correct / total if total > 0 else 0.0


def defined_in_partial_match(predictions: List[str], references: List[str]) -> float:
    """
    Calculate partial match accuracy for DefinedIn task (filename-only match).

    Args:
        predictions: List of model predictions
        references: List of reference JSON strings

    Returns:
        Partial match accuracy (0.0 to 1.0)
    """
    correct = 0
    total = len(predictions)

    for pred, ref in zip(predictions, references):
        try:
            pred_json = _parse_model_output(pred, 'DefinedIn')
            ref_json = json.loads(ref)

            # Check if filename matches (most important field)
            if pred_json.get("filename") == ref_json.get("filename"):
                correct += 1

        except (json.JSONDecodeError, AttributeError):
            # Prediction couldn't be parsed as JSON
            pass

    return correct / total if total > 0 else 0.0


def belongs_to_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Calculate exact match accuracy for BelongsTo task.

    Args:
        predictions: List of model predictions
        references: List of reference JSON strings

    Returns:
        Exact match accuracy (0.0 to 1.0)
    """
    correct = 0
    total = len(predictions)

    for pred, ref in zip(predictions, references):
        try:
            pred_json = _parse_model_output(pred, 'BelongsTo')
            ref_json = json.loads(ref)

            # Check if all fields match exactly (filename, class, method)
            if (pred_json.get("filename") == ref_json.get("filename") and
                pred_json.get("class") == ref_json.get("class") and
                pred_json.get("method") == ref_json.get("method")):
                correct += 1

        except (json.JSONDecodeError, AttributeError):
            # Prediction couldn't be parsed as JSON
            pass

    return correct / total if total > 0 else 0.0


def belongs_to_partial_match(predictions: List[str], references: List[str]) -> float:
    """
    Calculate partial match accuracy for BelongsTo task (filename-only match).

    Args:
        predictions: List of model predictions
        references: List of reference JSON strings

    Returns:
        Partial match accuracy (0.0 to 1.0)
    """
    correct = 0
    total = len(predictions)

    for pred, ref in zip(predictions, references):
        try:
            pred_json = _parse_model_output(pred, 'BelongsTo')
            ref_json = json.loads(ref)

            # Check if filename matches (most important field)
            if pred_json.get("filename") == ref_json.get("filename"):
                correct += 1

        except (json.JSONDecodeError, AttributeError):
            # Prediction couldn't be parsed as JSON
            pass

    return correct / total if total > 0 else 0.0


def calls_what_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Calculate exact match accuracy for CallsWhat task.
    Score = (number of correctly predicted calls) / (total ground truth calls)

    Args:
        predictions: List of model predictions
        references: List of reference JSON strings

    Returns:
        Average ratio of correctly predicted calls (0.0 to 1.0)
    """
    total_scores = []

    for pred, ref in zip(predictions, references):
        try:
            pred_json = _parse_model_output(pred, 'CallsWhat')
            ref_json = json.loads(ref)

            pred_calls = pred_json.get("calls", [])
            ref_calls = ref_json.get("calls", [])

            if not ref_calls:
                # If no ground truth calls, score is 1.0 if prediction is also empty
                score = 1.0 if not pred_calls else 0.0
            else:
                # Count how many ground truth calls are correctly predicted (exact match)
                correct_count = 0
                for ref_call in ref_calls:
                    for pred_call in pred_calls:
                        if (pred_call.get("filename") == ref_call.get("filename") and
                            pred_call.get("class") == ref_call.get("class") and
                            pred_call.get("method") == ref_call.get("method")):
                            correct_count += 1
                            break

                score = correct_count / len(ref_calls)

            total_scores.append(score)

        except (json.JSONDecodeError, AttributeError, TypeError):
            # Prediction couldn't be parsed as JSON or has wrong structure
            total_scores.append(0.0)

    return sum(total_scores) / len(total_scores) if total_scores else 0.0


def calls_what_partial_match(predictions: List[str], references: List[str]) -> float:
    """
    Calculate partial match accuracy for CallsWhat task.
    Score = (number of correctly predicted calls) / (total ground truth calls)
    Partial match: only checks if ground truth method names exactly match predicted method names (ignoring filename/class)

    Args:
        predictions: List of model predictions
        references: List of reference JSON strings

    Returns:
        Average ratio of correctly predicted calls (0.0 to 1.0)
    """
    total_scores = []

    for pred, ref in zip(predictions, references):
        try:
            pred_json = _parse_model_output(pred, 'CallsWhat')
            ref_json = json.loads(ref)

            pred_calls = pred_json.get("calls", [])
            ref_calls = ref_json.get("calls", [])

            if not ref_calls:
                # If no ground truth calls, score is 1.0 if prediction is also empty
                score = 1.0 if not pred_calls else 0.0
            else:
                # Count how many ground truth calls have their method names exactly matched (ignoring filename/class)
                correct_count = 0
                for ref_call in ref_calls:
                    ref_method = ref_call.get("method", "").strip()
                    if ref_method:
                        # Check if the reference method name exactly matches any predicted call method
                        for pred_call in pred_calls:
                            pred_method = pred_call.get("method", "").strip()
                            if ref_method == pred_method:
                                correct_count += 1
                                break

                score = correct_count / len(ref_calls)

            total_scores.append(score)

        except (json.JSONDecodeError, AttributeError, TypeError):
            # Prediction couldn't be parsed as JSON or has wrong structure
            total_scores.append(0.0)

    return sum(total_scores) / len(total_scores) if total_scores else 0.0


def called_by_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Calculate exact match accuracy for CalledBy task.
    Score = (number of correctly predicted callers) / (total ground truth callers)

    Args:
        predictions: List of model predictions
        references: List of reference JSON strings

    Returns:
        Average ratio of correctly predicted callers (0.0 to 1.0)
    """
    total_scores = []

    for pred, ref in zip(predictions, references):
        try:
            pred_json = _parse_model_output(pred, 'CalledBy')
            ref_json = json.loads(ref)

            pred_callers = pred_json.get("callers", [])
            ref_callers = ref_json.get("callers", [])

            if not ref_callers:
                # If no ground truth callers, score is 1.0 if prediction is also empty
                score = 1.0 if not pred_callers else 0.0
            else:
                # Count how many ground truth callers are correctly predicted (exact match)
                correct_count = 0
                for ref_caller in ref_callers:
                    for pred_caller in pred_callers:
                        if (pred_caller.get("filename") == ref_caller.get("filename") and
                            pred_caller.get("class") == ref_caller.get("class") and
                            pred_caller.get("method") == ref_caller.get("method")):
                            correct_count += 1
                            break

                score = correct_count / len(ref_callers)

            total_scores.append(score)

        except (json.JSONDecodeError, AttributeError, TypeError):
            # Prediction couldn't be parsed as JSON or has wrong structure
            total_scores.append(0.0)

    return sum(total_scores) / len(total_scores) if total_scores else 0.0


def called_by_partial_match(predictions: List[str], references: List[str]) -> float:
    """
    Calculate partial match accuracy for CalledBy task.
    Score = (number of correctly predicted callers) / (total ground truth callers)
    Partial match: only checks if ground truth method names exactly match predicted method names (ignoring filename/class)

    Args:
        predictions: List of model predictions
        references: List of reference JSON strings

    Returns:
        Average ratio of correctly predicted callers (0.0 to 1.0)
    """
    total_scores = []

    for pred, ref in zip(predictions, references):
        try:
            pred_json = _parse_model_output(pred, 'CalledBy')
            ref_json = json.loads(ref)

            pred_callers = pred_json.get("callers", [])
            ref_callers = ref_json.get("callers", [])

            if not ref_callers:
                # If no ground truth callers, score is 1.0 if prediction is also empty
                score = 1.0 if not pred_callers else 0.0
            else:
                # Count how many ground truth callers have their method names exactly matched (ignoring filename/class)
                correct_count = 0
                for ref_caller in ref_callers:
                    ref_method = ref_caller.get("method", "").strip()
                    if ref_method:
                        # Check if the reference caller method name exactly matches any predicted caller method
                        for pred_caller in pred_callers:
                            pred_method = pred_caller.get("method", "").strip()
                            if ref_method == pred_method:
                                correct_count += 1
                                break

                score = correct_count / len(ref_callers)

            total_scores.append(score)

        except (json.JSONDecodeError, AttributeError, TypeError):
            # Prediction couldn't be parsed as JSON or has wrong structure
            total_scores.append(0.0)

    return sum(total_scores) / len(total_scores) if total_scores else 0.0


# Register metrics exactly once to prevent duplicate-registration assertions when
# YAML task configs import this module multiple times via spec_from_file_location
if "defined_in_exact_match" not in registry.METRIC_REGISTRY:
    register_metric(metric="defined_in_exact_match", higher_is_better=True)(defined_in_exact_match)

if "defined_in_partial_match" not in registry.METRIC_REGISTRY:
    register_metric(metric="defined_in_partial_match", higher_is_better=True)(defined_in_partial_match)

if "belongs_to_exact_match" not in registry.METRIC_REGISTRY:
    register_metric(metric="belongs_to_exact_match", higher_is_better=True)(belongs_to_exact_match)

if "belongs_to_partial_match" not in registry.METRIC_REGISTRY:
    register_metric(metric="belongs_to_partial_match", higher_is_better=True)(belongs_to_partial_match)

if "calls_what_exact_match" not in registry.METRIC_REGISTRY:
    register_metric(metric="calls_what_exact_match", higher_is_better=True)(calls_what_exact_match)

if "calls_what_partial_match" not in registry.METRIC_REGISTRY:
    register_metric(metric="calls_what_partial_match", higher_is_better=True)(calls_what_partial_match)

if "called_by_exact_match" not in registry.METRIC_REGISTRY:
    register_metric(metric="called_by_exact_match", higher_is_better=True)(called_by_exact_match)

if "called_by_partial_match" not in registry.METRIC_REGISTRY:
    register_metric(metric="called_by_partial_match", higher_is_better=True)(called_by_partial_match)


def loc_file_recall(predictions: List[str], references: List[str]) -> float:
    """
    Simple recall-style metric for LocFile.

    For each example, count how many ground-truth file paths are present
    in the predicted list of files. Score per example = matches / |GT|.

    - Predictions are parsed from fenced JSON blocks:
      {"localized_files": ["xarray/core/dataset.py", ...]}
    - References are JSON strings with the same schema (expected_outputs_json).
    """
    scores = []

    for pred, ref in zip(predictions, references):
        try:
            # LM-eval may pass nested lists; normalize to string
            pred_str = str(pred[0]) if isinstance(pred, list) and pred else str(pred)

            # Extract last fenced JSON block (case-insensitive) and read files
            pred_json = extract_fenced_json(pred_str) or {}
            pred_files = set(map(str, pred_json.get("localized_files", [])))

            ref_json = json.loads(ref)
            gt_files = list(map(str, ref_json.get("localized_files", [])))

            if not gt_files:
                # If no GT, give full credit only when prediction is also empty
                score = 1.0 if not pred_files else 0.0
            else:
                matches = sum(1 for f in gt_files if f in pred_files)
                score = matches / len(gt_files)

            scores.append(score)
        except Exception:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


# Register new Loc* metric(s)
if "loc_file_recall" not in registry.METRIC_REGISTRY:
    register_metric(metric="loc_file_recall", higher_is_better=True)(loc_file_recall)


def loc_file_precision(predictions: List[str], references: List[str]) -> float:
    """
    Precision-style metric for LocFile.

    For each example, count how many predicted file paths are correct and divide
    by the number of predicted paths. Empty predictions receive 0.0 so silence
    is penalized when ground truth exists.
    """
    scores = []

    for pred, ref in zip(predictions, references):
        try:
            pred_str = str(pred[0]) if isinstance(pred, list) and pred else str(pred)
            pred_json = extract_fenced_json(pred_str) or {}
            pred_files = set(map(str, pred_json.get("localized_files", [])))

            ref_json = json.loads(ref)
            gt_files = set(map(str, ref_json.get("localized_files", [])))

            if not pred_files:
                scores.append(0.0)
                continue

            matches = sum(1 for f in pred_files if f in gt_files)
            score = matches / len(pred_files) if pred_files else 0.0
            scores.append(score)
        except Exception:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def loc_func_recall(predictions: List[str], references: List[str]) -> float:
    """
    Simple recall metric for LocFunc.

    For each ground-truth function entry (filename, symbol), award a point if an
    identical (filename, symbol) pair appears in predictions. Score per example
    = matches / |GT|. Ignores symbol_type for matching.
    """
    scores = []

    for pred, ref in zip(predictions, references):
        try:
            pred_str = str(pred[0]) if isinstance(pred, list) and pred else str(pred)
            pred_json = extract_fenced_json(pred_str) or {}
            pred_funcs = pred_json.get("localized_functions", []) or []

            # Deduplicate by (filename, symbol)
            pred_pairs = {
                (str(it.get("filename", "")), str(it.get("symbol", "")))
                for it in pred_funcs if isinstance(it, dict)
            }

            ref_json = json.loads(ref)
            ref_funcs = ref_json.get("localized_functions", []) or []
            gt_pairs = [
                (str(it.get("filename", "")), str(it.get("symbol", "")))
                for it in ref_funcs if isinstance(it, dict)
            ]

            if not gt_pairs:
                score = 1.0 if not pred_pairs else 0.0
            else:
                matches = sum(1 for p in gt_pairs if p in pred_pairs)
                score = matches / len(gt_pairs)

            scores.append(score)
        except Exception:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def _expand_range_to_lines(range_str: str) -> Set[int]:
    """Expand a range string like '5-7' or '42' into a set of integers."""
    if range_str is None:
        return set()
    cleaned = str(range_str).strip()
    if not cleaned:
        return set()
    cleaned = cleaned.replace(" ", "")
    if "-" in cleaned:
        parts = cleaned.split("-")
        if len(parts) != 2:
            return set()
        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError:
            return set()
        if start > end:
            start, end = end, start
        return set(range(start, end + 1))
    try:
        value = int(cleaned)
    except ValueError:
        return set()
    return {value}


def _lines_by_file(obj: dict) -> Dict[str, Set[int]]:
    """Collect expanded line numbers per filename from a LocLine JSON payload."""
    result: Dict[str, Set[int]] = {}
    for item in (obj.get("localized_spans", []) or []):
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename", "")).strip()
        if not filename:
            continue
        ranges = item.get("localized_line_ranges", []) or []
        for rng in ranges:
            for line in _expand_range_to_lines(rng):
                result.setdefault(filename, set()).add(line)
    return result


def loc_line_recall(predictions: List[str], references: List[str]) -> float:
    """
    Recall metric for LocLine using expanded line numbers.

    Each range string (e.g., "10-12", "45") is expanded into individual line
    numbers per file, and recall is computed as matches / |GT lines|.

    Notes:
    - Ranges are inclusive (e.g., "2-4" -> {2,3,4}).
    - Invalid or empty ranges are ignored.
    - 'localized_components' is ignored for scoring.
    """
    def expand_lines_by_file(obj: dict):
        result = {}
        for item in (obj.get("localized_spans", []) or []):
            if not isinstance(item, dict):
                continue
            filename = str(item.get("filename", "")).strip()
            if not filename:
                continue
            ranges = item.get("localized_line_ranges", []) or []
            for rng in ranges:
                for line in _expand_range_to_lines(str(rng)):
                    result.setdefault(filename, set()).add(line)
        return result

    scores = []

    for pred, ref in zip(predictions, references):
        try:
            pred_str = str(pred[0]) if isinstance(pred, list) and pred else str(pred)
            pred_json = extract_fenced_json(pred_str) or {}
            pred_lines = expand_lines_by_file(pred_json)

            ref_json = json.loads(ref)
            gt_lines = expand_lines_by_file(ref_json)

            total_gt = sum(len(lines) for lines in gt_lines.values())
            total_pred = sum(len(lines) for lines in pred_lines.values())

            if total_gt == 0:
                score = 1.0 if total_pred == 0 else 0.0
            else:
                matches = 0
                for filename, gt_set in gt_lines.items():
                    matches += len(gt_set & pred_lines.get(filename, set()))
                score = matches / total_gt if total_gt else 0.0

            scores.append(score)
        except Exception:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


if "loc_func_recall" not in registry.METRIC_REGISTRY:
    register_metric(metric="loc_func_recall", higher_is_better=True)(loc_func_recall)

if "loc_line_recall" not in registry.METRIC_REGISTRY:
    register_metric(metric="loc_line_recall", higher_is_better=True)(loc_line_recall)


def loc_func_precision(predictions: List[str], references: List[str]) -> float:
    """
    Precision metric for LocFunc using (filename, symbol) pairs.
    """
    scores = []

    for pred, ref in zip(predictions, references):
        try:
            pred_str = str(pred[0]) if isinstance(pred, list) and pred else str(pred)
            pred_json = extract_fenced_json(pred_str) or {}
            pred_funcs = pred_json.get("localized_functions", []) or []

            pred_pairs = {
                (str(it.get("filename", "")), str(it.get("symbol", "")))
                for it in pred_funcs if isinstance(it, dict)
            }

            ref_json = json.loads(ref)
            ref_funcs = ref_json.get("localized_functions", []) or []
            gt_pairs = {
                (str(it.get("filename", "")), str(it.get("symbol", "")))
                for it in ref_funcs if isinstance(it, dict)
            }

            if not pred_pairs:
                scores.append(0.0)
                continue

            matches = sum(1 for p in pred_pairs if p in gt_pairs)
            score = matches / len(pred_pairs) if pred_pairs else 0.0
            scores.append(score)
        except Exception:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def loc_line_precision(predictions: List[str], references: List[str]) -> float:
    """
    Precision metric for LocLine using expanded line numbers per file.
    """
    def expand_lines_by_file(obj: dict):
        result = {}
        for item in (obj.get("localized_spans", []) or []):
            if not isinstance(item, dict):
                continue
            filename = str(item.get("filename", "")).strip()
            if not filename:
                continue
            ranges = item.get("localized_line_ranges", []) or []
            for rng in ranges:
                for line in _expand_range_to_lines(str(rng)):
                    result.setdefault(filename, set()).add(line)
        return result

    scores = []

    for pred, ref in zip(predictions, references):
        try:
            pred_str = str(pred[0]) if isinstance(pred, list) and pred else str(pred)
            pred_json = extract_fenced_json(pred_str) or {}
            pred_lines = expand_lines_by_file(pred_json)

            ref_json = json.loads(ref)
            gt_lines = expand_lines_by_file(ref_json)

            total_pred = sum(len(lines) for lines in pred_lines.values())
            if total_pred == 0:
                scores.append(0.0)
                continue

            matches = 0
            for filename, pred_set in pred_lines.items():
                matches += len(pred_set & gt_lines.get(filename, set()))

            score = matches / total_pred if total_pred else 0.0
            scores.append(score)
        except Exception:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


if "loc_file_precision" not in registry.METRIC_REGISTRY:
    register_metric(metric="loc_file_precision", higher_is_better=True)(loc_file_precision)

if "loc_func_precision" not in registry.METRIC_REGISTRY:
    register_metric(metric="loc_func_precision", higher_is_better=True)(loc_func_precision)

if "loc_line_precision" not in registry.METRIC_REGISTRY:
    register_metric(metric="loc_line_precision", higher_is_better=True)(loc_line_precision)


# ---------------------------------------------------------------------------
# CodeEdit metrics (PatchSim)
# ---------------------------------------------------------------------------

def _norm_replace(text: str) -> str:
    """Normalize replacement text for similarity scoring."""
    if text is None:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))

    out_lines: List[str] = []
    blank = False
    for line in normalized.split("\n"):
        if line == "":
            if not blank:
                out_lines.append("")
            blank = True
        else:
            out_lines.append(line)
            blank = False

    return "\n".join(out_lines)


def _code_edit_similarity(a: str, b: str) -> float:
    """Return difflib-based similarity between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def _normalize_code_edits(edits: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Filter and normalize code edit entries."""
    normalized: List[Dict[str, str]] = []
    for entry in edits or []:
        if not isinstance(entry, dict):
            continue
        file_path = str(entry.get("file", "")).strip()
        if not file_path:
            continue
        search = entry.get("search")
        if search is None:
            continue
        search_str = str(search)
        replace = entry.get("replace", "")
        replace_str = str(replace) if replace is not None else ""
        normalized.append(
            {
                "file": file_path,
                "search": search_str,
                "replace": replace_str,
            }
        )
    return normalized


def _code_edit_components(
    gt_ops: List[Dict[str, str]],
    pred_ops: List[Dict[str, str]],
) -> Dict[str, float]:
    """Compute per-example PatchSim components."""
    gold_map = {(op["file"], op["search"]): op["replace"] for op in gt_ops}
    pred_map = {(op["file"], op["search"]): op["replace"] for op in pred_ops}

    gold_keys = set(gold_map)
    pred_keys = set(pred_map)
    matched_keys = gold_keys & pred_keys

    gold_count = len(gold_keys)
    pred_count = len(pred_keys)
    match_count = len(matched_keys)

    degenerate = int(gold_count == 0 and pred_count == 0)

    if matched_keys:
        sim_num = 0.0
        sim_den = 0.0
        for key in matched_keys:
            gold_text = _norm_replace(gold_map[key])
            pred_text = _norm_replace(pred_map[key])
            weight = max(len(gold_text), 1)  # avoid zero division for empty strings
            sim_num += weight * _code_edit_similarity(pred_text, gold_text)
            sim_den += weight
    else:
        sim_num = 0.0
        sim_den = 0.0

    return {
        "gold": float(gold_count),
        "pred": float(pred_count),
        "match": float(match_count),
        "sim_num": sim_num,
        "sim_den": sim_den,
        "degenerate": float(degenerate),
    }


def _aggregate_code_edit(items: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate PatchSim components across dataset."""
    total_gold = sum(item.get("gold", 0.0) for item in items)
    total_pred = sum(item.get("pred", 0.0) for item in items)
    total_match = sum(item.get("match", 0.0) for item in items)
    total_sim_num = sum(item.get("sim_num", 0.0) for item in items)
    total_sim_den = sum(item.get("sim_den", 0.0) for item in items)
    degenerate = int(sum(item.get("degenerate", 0.0) for item in items))
    item_count = len(items)

    if item_count == 0:
        return {"key_f1": 0.0, "replace_sim": 0.0, "patchsim": 0.0}

    if total_pred == 0 and total_gold == 0 and degenerate == item_count:
        # No edits anywhere; perfect score
        return {"key_f1": 1.0, "replace_sim": 1.0, "patchsim": 1.0}

    key_prec = total_match / total_pred if total_pred > 0 else 0.0
    key_rec = total_match / total_gold if total_gold > 0 else 0.0
    key_f1 = (2 * key_prec * key_rec) / (key_prec + key_rec) if (key_prec + key_rec) else 0.0

    if total_sim_den > 0:
        replace_sim = total_sim_num / total_sim_den
    elif degenerate == item_count:
        replace_sim = 1.0
    else:
        replace_sim = 0.0

    patchsim = key_f1 * replace_sim
    return {"key_f1": key_f1, "replace_sim": replace_sim, "patchsim": patchsim}


def code_edit_metric_components(predictions: List[str], references: List[str]) -> List[Dict[str, float]]:
    """Return per-example PatchSim components for aggregation."""
    items: List[Dict[str, float]] = []

    for pred, ref in zip(predictions, references):
        try:
            pred_str = str(pred[0]) if isinstance(pred, list) and pred else str(pred)
            parsed_pred = _parse_model_output(pred_str, "CodeEdit")
            pred_edits = _normalize_code_edits(parsed_pred.get("code_edits", []))

            ref_json = json.loads(ref)
            gt_edits = _normalize_code_edits(ref_json.get("code_edits", []))

            components = _code_edit_components(gt_edits, pred_edits)
        except Exception:
            components = _code_edit_components([], [])

        items.append(components)

    return items


@register_aggregation("code_edit_patchsim_micro")
def code_edit_patchsim_micro(items: List[Dict[str, float]]) -> float:
    return _aggregate_code_edit(items)["patchsim"]


@register_aggregation("code_edit_keyf1_micro")
def code_edit_keyf1_micro(items: List[Dict[str, float]]) -> float:
    return _aggregate_code_edit(items)["key_f1"]


@register_aggregation("code_edit_replacesim_micro")
def code_edit_replacesim_micro(items: List[Dict[str, float]]) -> float:
    return _aggregate_code_edit(items)["replace_sim"]


def _register_code_edit_metrics():
    if "code_edit_patchsim" not in registry.METRIC_REGISTRY:
        register_metric(
            metric="code_edit_patchsim",
            higher_is_better=True,
            aggregation="code_edit_patchsim_micro",
        )(code_edit_metric_components)

    if "code_edit_keyf1" not in registry.METRIC_REGISTRY:
        register_metric(
            metric="code_edit_keyf1",
            higher_is_better=True,
            aggregation="code_edit_keyf1_micro",
        )(code_edit_metric_components)

    if "code_edit_replacesim" not in registry.METRIC_REGISTRY:
        register_metric(
            metric="code_edit_replacesim",
            higher_is_better=True,
            aggregation="code_edit_replacesim_micro",
        )(code_edit_metric_components)


_register_code_edit_metrics()
