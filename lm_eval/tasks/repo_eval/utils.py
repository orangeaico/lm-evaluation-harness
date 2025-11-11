"""
Utilities for repo-eval tasks in lm-evaluation-harness.

This module provides data processing and metric calculation functions
for the DefinedIn, BelongsTo, CallsWhat, and CalledBy code understanding tasks.
Enhanced with fenced JSON parsing and Pydantic validation.
"""

import copy
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import datasets
from lm_eval.api import registry
from lm_eval.api.metrics import aggregate_subtask_metrics
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
    parse_test_loc_file,
    parse_test_code_edit,
    extract_fenced_json,
)
from eval.parsers.str_replace_parser import convert_str_replace_to_code_edits


TASK_PARSERS = {
    'DefinedIn': parse_defined_in,
    'BelongsTo': parse_belongs_to,
    'CallsWhat': parse_calls_what,
    'CalledBy': parse_called_by,
    'LocFile': parse_loc_file,
    'LocFunc': parse_loc_func,
    'LocLine': parse_loc_line,
    'CodeEdit': parse_code_edit,
    'TestLocFile': parse_test_loc_file,
    'TestCodeEdit': parse_test_code_edit,
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


def process_test_loc_file_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process TestLocFile documents by normalizing to LocFile schema."""

    def _process_doc(doc):
        expected_outputs = copy.deepcopy(doc["expected_outputs"])
        test_files = expected_outputs.get("test_files")
        if isinstance(test_files, list):
            expected_outputs["localized_files"] = test_files
        expected_outputs_json = json.dumps(expected_outputs, sort_keys=True)
        return {
            "prompt": doc["prompt"],
            "expected_outputs": expected_outputs,
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
        raw_expected = copy.deepcopy(doc["expected_outputs"])
        if isinstance(raw_expected, str):
            expected_outputs = {"code_edits": convert_str_replace_to_code_edits(raw_expected)}
        elif isinstance(raw_expected, dict):
            code_edits = raw_expected.get("code_edits")
            if isinstance(code_edits, str):
                raw_expected = dict(raw_expected)
                raw_expected["code_edits"] = convert_str_replace_to_code_edits(code_edits)
            expected_outputs = raw_expected
        else:
            expected_outputs = {"code_edits": []}

        expected_outputs["_prompt"] = doc["prompt"]
        expected_outputs_json = json.dumps(expected_outputs, sort_keys=True)
        expected_outputs.pop("_prompt", None)
        return {
            "prompt": doc["prompt"],
            "expected_outputs": expected_outputs,
            "expected_outputs_json": expected_outputs_json,
            "task_id": doc["task_id"],
            "repo_name": doc["repo_name"],
            "commit_hash": doc["commit_hash"],
            "difficulty_level": doc.get("difficulty_level", "unknown"),
        }

    return dataset.map(_process_doc)


def process_test_code_edit_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process TestCodeEdit documents for lm-eval format."""

    def _process_doc(doc):
        raw_expected = copy.deepcopy(doc["expected_outputs"])
        expected_outputs: Dict[str, Any]
        if isinstance(raw_expected, str):
            edits = convert_str_replace_to_code_edits(raw_expected)
            expected_outputs = {
                "code_edits": edits,
                "test_code_edits": edits,
            }
        elif isinstance(raw_expected, dict):
            expected_outputs = dict(raw_expected)
            test_code_edits = expected_outputs.get("test_code_edits")
            code_edits = expected_outputs.get("code_edits")
            if isinstance(test_code_edits, str):
                edits = convert_str_replace_to_code_edits(test_code_edits)
                expected_outputs["test_code_edits"] = edits
                expected_outputs.setdefault("code_edits", edits)
            elif isinstance(test_code_edits, list) and not code_edits:
                expected_outputs["code_edits"] = test_code_edits
            elif code_edits and not expected_outputs.get("test_code_edits"):
                expected_outputs["test_code_edits"] = code_edits
        else:
            expected_outputs = {"code_edits": [], "test_code_edits": []}

        expected_outputs["_prompt"] = doc["prompt"]
        expected_outputs_json = json.dumps(expected_outputs, sort_keys=True)
        expected_outputs.pop("_prompt", None)
        return {
            "prompt": doc["prompt"],
            "expected_outputs": expected_outputs,
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
            pred_files = pred_json.get("localized_files")
            if not pred_files:
                pred_files = pred_json.get("test_files", [])
            pred_files = set(map(str, pred_files))

            ref_json = json.loads(ref)
            gt_files = ref_json.get("localized_files")
            if not gt_files:
                gt_files = ref_json.get("test_files", [])
            gt_files = list(map(str, gt_files))

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
            pred_files = pred_json.get("localized_files")
            if not pred_files:
                pred_files = pred_json.get("test_files", [])
            pred_files = set(map(str, pred_files))

            ref_json = json.loads(ref)
            gt_files = ref_json.get("localized_files")
            if not gt_files:
                gt_files = ref_json.get("test_files", [])
            gt_files = set(map(str, gt_files))

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


def _sanitize_code_edits(edits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter, normalize, and deduplicate code edit entries per file."""
    initial: List[Dict[str, Any]] = []

    for idx, entry in enumerate(edits or []):
        if not isinstance(entry, dict):
            continue
        file_path = str(entry.get("file", "")).strip()
        if not file_path:
            continue
        search = entry.get("search")
        if search is None:
            continue
        search_str = str(search)
        if not search_str:
            continue
        replace = entry.get("replace", "")
        replace_str = str(replace) if replace is not None else ""
        initial.append(
            {
                "file": file_path,
                "search": search_str,
                "replace": replace_str,
                "_orig_index": idx,
            }
        )

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in initial:
        grouped.setdefault(item["file"], []).append(item)

    sanitized: List[Dict[str, Any]] = []
    for file_path, items in grouped.items():
        filtered: List[Dict[str, Any]] = []
        for candidate in sorted(items, key=lambda it: (-len(it["search"]), it["_orig_index"])):
            if any(candidate["search"] in kept["search"] for kept in filtered):
                continue
            filtered.append(candidate)
        sanitized.extend(sorted(filtered, key=lambda it: it["_orig_index"]))

    sanitized.sort(key=lambda it: (it["file"], it["_orig_index"]))
    return sanitized


def _find_available_span(
    prompt: str, snippet: str, used: List[Tuple[int, int]]
) -> Tuple[int | None, int | None]:
    """Locate a snippet within the prompt without reusing occupied spans."""
    if not prompt or not snippet:
        return None, None

    start = prompt.find(snippet)
    while start != -1:
        end = start + len(snippet)
        if not any(not (end <= u_start or start >= u_end) for u_start, u_end in used):
            used.append((start, end))
            return start, end
        start = prompt.find(snippet, start + 1)

    return None, None


def _prepare_code_edit_entries(
    edits: List[Dict[str, Any]], prompt: str
) -> List[Dict[str, Any]]:
    """Sanitize edits and enrich them with ordering metadata."""
    sanitized = _sanitize_code_edits(edits)
    prepared: List[Dict[str, Any]] = []
    used_spans: List[Tuple[int, int]] = []
    fallback_counter = 0

    for item in sanitized:
        start, end = _find_available_span(prompt, item["search"], used_spans)
        order_key = start if start is not None else (10**12 + fallback_counter)
        fallback_counter += 1

        prepared.append(
            {
                "file": item["file"],
                "search": item["search"],
                "replace": item["replace"],
                "start": start,
                "end": end,
                "_order_key": order_key,
                "_orig_index": item["_orig_index"],
            }
        )

    prepared.sort(key=lambda it: (it["_order_key"], it["_orig_index"]))
    return prepared


def _build_code_edit_patch(entries: List[Dict[str, Any]]) -> str:
    """Create an ordered textual patch representation for similarity scoring."""
    lines: List[str] = []
    for entry in entries:
        start = entry.get("start")
        end = entry.get("end")
        if start is None or end is None:
            header = f"@@ {entry['file']} @@"
        else:
            header = f"@@ {entry['file']} {start}-{end} @@"
        lines.append(header)
        lines.append("-" + entry["search"])
        lines.append("+" + entry["replace"])
    return "\n".join(lines)


def _span_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    """Return the number of overlapping characters between two spans."""
    if a.get("start") is None or b.get("start") is None:
        return 0
    start = max(a["start"], b["start"])
    end = min(a["end"], b["end"])
    return max(0, end - start)


def _match_code_edit_entries(
    gold_entries: List[Dict[str, Any]],
    pred_entries: List[Dict[str, Any]],
) -> int:
    """Pair predicted edits with gold edits using similarity and span overlap."""
    matches = 0
    used_pred: Set[int] = set()

    for gold in gold_entries:
        best_idx = None
        best_score = 0.0

        for idx, pred in enumerate(pred_entries):
            if idx in used_pred:
                continue
            if pred["file"] != gold["file"]:
                continue

            similarity = _code_edit_similarity(pred["search"], gold["search"])
            if not similarity and not _span_overlap(gold, pred):
                continue

            if gold["search"] in pred["search"] or pred["search"] in gold["search"]:
                similarity = max(similarity, 0.7 if gold["search"] != pred["search"] else 1.0)

            overlap_bonus = 1.0 if _span_overlap(gold, pred) > 0 else 0.0
            score = similarity + overlap_bonus

            if score > best_score and similarity >= 0.5:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            used_pred.add(best_idx)
            matches += 1

    return matches


def _code_edit_components(
    gt_ops: List[Dict[str, Any]],
    pred_ops: List[Dict[str, Any]],
    prompt: str,
) -> Dict[str, float]:
    """Compute per-example PatchSim components using ordered patch comparison."""
    gold_entries = _prepare_code_edit_entries(gt_ops, prompt)
    pred_entries = _prepare_code_edit_entries(pred_ops, prompt)

    gold_count = len(gold_entries)
    pred_count = len(pred_entries)
    match_count = _match_code_edit_entries(gold_entries, pred_entries)
    degenerate = int(gold_count == 0 and pred_count == 0)

    gold_patch = _build_code_edit_patch(gold_entries)
    pred_patch = _build_code_edit_patch(pred_entries)

    if gold_patch or pred_patch:
        weight = max(len(gold_patch), 1)
        similarity = SequenceMatcher(None, pred_patch, gold_patch).ratio()
        sim_num = similarity * weight
        sim_den = weight
    else:
        sim_num = 0.0
        sim_den = 0.0

    if degenerate:
        sim_num = 0.0
        sim_den = 0.0

    return {
        "gold": float(gold_count),
        "pred": float(pred_count),
        "match": float(match_count),
        "sim_num": float(sim_num),
        "sim_den": float(sim_den),
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

def _flatten_code_edit_items(items: List[Any]) -> List[Dict[str, float]]:
    """Normalize raw metric outputs into a list of component dicts."""
    flattened: List[Dict[str, float]] = []
    for entry in items or []:
        if isinstance(entry, dict):
            flattened.append(entry)
        elif isinstance(entry, (list, tuple)):
            for sub_entry in entry:
                if isinstance(sub_entry, dict):
                    flattened.append(sub_entry)
        elif hasattr(entry, "items"):
            flattened.append(dict(entry))
    return flattened


def _aggregate_code_edit_value(items: List[Any], key: str) -> float:
    """Aggregate a specific PatchSim component across dataset outputs."""
    components = _flatten_code_edit_items(items)
    aggregated = _aggregate_code_edit(components)
    return float(aggregated.get(key, 0.0))


def code_edit_patchsim_macro(items: List[Any]) -> float:
    """Macro average PatchSim across a single dataset."""
    return _aggregate_code_edit_value(items, "patchsim")


def code_edit_keyf1_macro(items: List[Any]) -> float:
    """Macro average key-level F1 across a single dataset."""
    return _aggregate_code_edit_value(items, "key_f1")


def code_edit_replacesim_macro(items: List[Any]) -> float:
    """Macro average replace similarity across a single dataset."""
    return _aggregate_code_edit_value(items, "replace_sim")


def _extract_group_metric(
    metrics: List[Any], sizes: List[int], key: str
) -> tuple[List[float], List[int]]:
    """Extract numeric values for group-level aggregation while filtering invalid entries."""
    extracted: List[float] = []
    filtered_sizes: List[int] = []

    for value, size in zip(metrics, sizes):
        if value is None or (isinstance(value, str) and value == "N/A"):
            continue

        if isinstance(value, dict):
            numeric = value.get(key, 0.0)
        elif isinstance(value, (list, tuple)) and value:
            inner = value[0]
            numeric = inner.get(key, 0.0) if isinstance(inner, dict) else inner
        else:
            numeric = value

        try:
            extracted.append(float(numeric))
            filtered_sizes.append(size)
        except (TypeError, ValueError):
            continue

    return extracted, filtered_sizes


def code_edit_patchsim_micro(
    metrics: List[Any], sizes: List[int], weight_by_size: bool = True
) -> float:
    """Aggregate PatchSim across subtasks using micro-averaging semantics."""
    values, filtered_sizes = _extract_group_metric(metrics, sizes, "patchsim")
    if not values or not filtered_sizes:
        return 0.0
    return aggregate_subtask_metrics(values, filtered_sizes, weight_by_size)


def code_edit_keyf1_micro(
    metrics: List[Any], sizes: List[int], weight_by_size: bool = True
) -> float:
    """Aggregate key-level F1 across subtasks using micro-averaging semantics."""
    values, filtered_sizes = _extract_group_metric(metrics, sizes, "key_f1")
    if not values or not filtered_sizes:
        return 0.0
    return aggregate_subtask_metrics(values, filtered_sizes, weight_by_size)


def code_edit_replacesim_micro(
    metrics: List[Any], sizes: List[int], weight_by_size: bool = True
) -> float:
    """Aggregate replace similarity across subtasks using micro-averaging semantics."""
    values, filtered_sizes = _extract_group_metric(metrics, sizes, "replace_sim")
    if not values or not filtered_sizes:
        return 0.0
    return aggregate_subtask_metrics(values, filtered_sizes, weight_by_size)


def code_edit_metric_components(predictions: List[str], references: List[str]) -> Dict[str, List[Dict[str, float]]]:
    """Return PatchSim components keyed by metric names for aggregation."""
    components: List[Dict[str, float]] = []

    for pred, ref in zip(predictions, references):
        prompt = ""
        gt_edits: List[Dict[str, Any]] = []
        pred_edits: List[Dict[str, Any]] = []
        task_key = "CodeEdit"

        try:
            ref_json = json.loads(ref)
            prompt = ref_json.get("_prompt", "")
            if ref_json.get("test_code_edits") is not None:
                task_key = "TestCodeEdit"
            gt_edits = ref_json.get("code_edits") or ref_json.get("test_code_edits", [])
        except Exception:
            gt_edits = []
        try:
            if isinstance(pred, list):
                pred_str = str(pred[0]) if pred else ""
            else:
                pred_str = str(pred)
            parsed_pred = _parse_model_output(pred_str, task_key)
            pred_edits = parsed_pred.get("code_edits", [])
        except Exception:
            pred_edits = []

        components.append(_code_edit_components(gt_edits, pred_edits, prompt))

    return {
        "code_edit_patchsim": components,
        "code_edit_keyf1": components,
        "code_edit_replacesim": components,
    }


def _register_code_edit_metrics():
    if "code_edit_patchsim_macro" not in registry.AGGREGATION_REGISTRY:
        register_aggregation("code_edit_patchsim_macro")(code_edit_patchsim_macro)

    if "code_edit_keyf1_macro" not in registry.AGGREGATION_REGISTRY:
        register_aggregation("code_edit_keyf1_macro")(code_edit_keyf1_macro)

    if "code_edit_replacesim_macro" not in registry.AGGREGATION_REGISTRY:
        register_aggregation("code_edit_replacesim_macro")(code_edit_replacesim_macro)

    if "code_edit_patchsim" not in registry.METRIC_REGISTRY:
        register_metric(
            metric="code_edit_patchsim",
            higher_is_better=True,
            aggregation="code_edit_patchsim_macro",
        )(code_edit_metric_components)

    if "code_edit_keyf1" not in registry.METRIC_REGISTRY:
        register_metric(
            metric="code_edit_keyf1",
            higher_is_better=True,
            aggregation="code_edit_keyf1_macro",
        )(code_edit_metric_components)

    if "code_edit_replacesim" not in registry.METRIC_REGISTRY:
        register_metric(
            metric="code_edit_replacesim",
            higher_is_better=True,
            aggregation="code_edit_replacesim_macro",
        )(code_edit_metric_components)


_register_code_edit_metrics()
