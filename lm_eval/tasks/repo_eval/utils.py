"""
Utilities for repo-eval tasks in lm-evaluation-harness.

This module provides data processing and metric calculation functions
for the DefinedIn and BelongsTo code understanding tasks.
"""

import json
import re
from typing import Dict, Any, List
import datasets


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

            # Check if all required fields match exactly
            if (pred_json.get("absolute_path") == ref_json.get("absolute_path") and
                pred_json.get("filename") == ref_json.get("filename") and
                pred_json.get("class") == ref_json.get("class")):
                correct += 1

        except (json.JSONDecodeError, AttributeError):
            # Prediction couldn't be parsed as JSON
            pass

    return correct / total if total > 0 else 0.0


def defined_in_partial_match(predictions: List[str], references: List[str]) -> float:
    """
    Calculate partial match accuracy for DefinedIn task (path-only match).

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

            # Check if absolute_path matches (most important field)
            if pred_json.get("absolute_path") == ref_json.get("absolute_path"):
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

            # Check if both fields match exactly
            if (pred_json.get("absolute_path") == ref_json.get("absolute_path") and
                pred_json.get("component_path") == ref_json.get("component_path")):
                correct += 1

        except (json.JSONDecodeError, AttributeError):
            # Prediction couldn't be parsed as JSON
            pass

    return correct / total if total > 0 else 0.0


def belongs_to_partial_match(predictions: List[str], references: List[str]) -> float:
    """
    Calculate partial match accuracy for BelongsTo task (path-only match).

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

            # Check if absolute_path matches (most important field)
            if pred_json.get("absolute_path") == ref_json.get("absolute_path"):
                correct += 1

        except (json.JSONDecodeError, AttributeError):
            # Prediction couldn't be parsed as JSON
            pass

    return correct / total if total > 0 else 0.0