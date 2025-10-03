"""
Utilities for repo-eval tasks in lm-evaluation-harness.

This module provides data processing and metric calculation functions
for the DefinedIn and BelongsTo code understanding tasks.
"""

import json
import re
from typing import Dict, Any, List
import datasets
from lm_eval.api import registry
from lm_eval.api.registry import register_metric


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


class ExtractJSONFilter:
    """Filter class to extract JSON from model responses."""

    def apply(self, resps: List[str], docs: List[Dict[str, Any]]) -> List[str]:
        """
        Apply JSON extraction to model responses.

        Args:
            resps: List of model responses
            docs: List of documents (unused)

        Returns:
            List of processed responses with extracted JSON
        """
        processed = []
        for resp in resps:
            # Handle both string and list responses
            if isinstance(resp, list):
                # Take the first response if it's a list
                resp_str = str(resp[0]) if resp else ""
            else:
                resp_str = str(resp)
            processed.append(self.extract_json_response(resp_str))
        return processed

    def extract_json_response(self, response: str) -> str:
        """
        Extract JSON from model response.

        Args:
            response: Raw model response

        Returns:
            Extracted JSON string or original response if no JSON found
        """
        # Try to find JSON in the response
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested JSON
            r'\{.*?\}',  # Basic JSON pattern
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    # Validate it's proper JSON
                    json.loads(match)
                    return match.strip()
                except json.JSONDecodeError:
                    continue

        # If no valid JSON found, return the response as-is
        return response.strip()


def extract_json_response():
    """Factory function for JSON extraction filter."""
    return ExtractJSONFilter()


def _parse_model_output(response: str) -> Dict[str, Any]:
    """
    Parse model output and extract JSON.

    Args:
        response: Model response string

    Returns:
        Parsed JSON dict or empty dict if parsing fails
    """
    try:
        # First try to parse the response directly
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from the response using the filter class
        filter_instance = ExtractJSONFilter()
        extracted = filter_instance.extract_json_response(response)
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
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
            pred_json = _parse_model_output(pred)
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
            pred_json = _parse_model_output(pred)
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
            pred_json = _parse_model_output(pred)
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
            pred_json = _parse_model_output(pred)
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
            pred_json = _parse_model_output(pred)
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
            pred_json = _parse_model_output(pred)
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
            pred_json = _parse_model_output(pred)
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
            pred_json = _parse_model_output(pred)
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
