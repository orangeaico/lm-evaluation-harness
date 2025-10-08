"""
Utilities for repo-eval tasks in lm-evaluation-harness.

This module provides data processing and metric calculation functions
for the DefinedIn, BelongsTo, CallsWhat, and CalledBy code understanding tasks.
Enhanced with fenced JSON parsing and Pydantic validation.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import datasets
from lm_eval.api import registry
from lm_eval.api.registry import register_metric

# Ensure project src directory is on the path for parser imports
SRC_ROOT = Path(__file__).resolve().parents[5]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval.parsers.fenced_json_parser import (
    parse_defined_in,
    parse_belongs_to,
    parse_calls_what,
    parse_called_by
)


TASK_PARSERS = {
    'DefinedIn': parse_defined_in,
    'BelongsTo': parse_belongs_to,
    'CallsWhat': parse_calls_what,
    'CalledBy': parse_called_by,
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
            task_name: One of 'DefinedIn', 'BelongsTo', 'CallsWhat', 'CalledBy'
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
