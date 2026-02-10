"""Shared utilities for TraceLLM project.

This module contains common functions used across multiple ensemble methods
to eliminate code duplication and ensure consistency.
"""

from typing import Tuple, Union, List
import numpy as np
import pandas as pd


def to_binary_label(value: Union[int, float, str]) -> int:
    """Convert various label formats to binary 0/1.

    Args:
        value: Label value (can be int, float, or string like "yes"/"no")

    Returns:
        Binary label: 1 or 0

    Examples:
        >>> to_binary_label("yes")
        1
        >>> to_binary_label(0)
        0
        >>> to_binary_label(1.0)
        1
    """
    try:
        v = int(float(str(value).strip()))
        return 1 if v == 1 else 0
    except (ValueError, AttributeError):
        return 0


def calculate_f2_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
) -> Tuple[float, float, float]:
    """Calculate recall, precision, and F2 score.

    F2 score emphasizes recall over precision (beta=2), which is appropriate
    for requirements traceability where missing links (false negatives) are
    more costly than false positives.

    Args:
        y_true: Ground truth binary labels (1=trace link exists, 0=no link)
        y_pred: Predicted binary labels

    Returns:
        Tuple of (recall, precision, f2_score)

    Note:
        This implementation is used consistently across all TraceLLM variants
        to ensure fair comparison.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    beta2 = 4.0
    denom = (beta2 * precision) + recall
    f2 = (1 + beta2) * precision * recall / denom if denom > 0 else 0.0

    return float(recall), float(precision), float(f2)


# Column name variations across different datasets
POSSIBLE_SRC_ID_COLS: List[str] = ["source_id", "source_ID", "UC"]
POSSIBLE_TGT_ID_COLS: List[str] = ["target_id", "target_ID", "TC", "ID"]


def normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize source_id and target_id column names across datasets.

    Different datasets use different naming conventions (e.g., UC/TC for
    EasyClinic, source_id/target_id for others). This normalizes to
    consistent 'source_id' and 'target_id' columns.

    Args:
        df: DataFrame with dataset-specific ID column names

    Returns:
        DataFrame with standardized 'source_id' and 'target_id' columns

    Raises:
        ValueError: If required ID columns not found in any expected format
    """
    df = df.copy()

    # Normalize source_id
    for col in POSSIBLE_SRC_ID_COLS:
        if col in df.columns:
            if col != "source_id":
                df.rename(columns={col: "source_id"}, inplace=True)
            break
    else:
        raise ValueError(
            f"Missing source ID column. Expected one of: {POSSIBLE_SRC_ID_COLS}"
        )

    # Normalize target_id
    for col in POSSIBLE_TGT_ID_COLS:
        if col in df.columns:
            if col != "target_id":
                df.rename(columns={col: "target_id"}, inplace=True)
            break
    else:
        raise ValueError(
            f"Missing target ID column. Expected one of: {POSSIBLE_TGT_ID_COLS}"
        )

    return df


# System role prompts for different traceability tasks
SYSTEM_ROLES = {
    "easyclinic_uc_tc": (
        "You are an expert in software traceability. "
        "You are given two artifacts from a healthcare system. "
        "(1) is a use case and (2) is a test case. "
        "Does (2) directly test (1)?\n\n"
        "Respond strictly in JSON format with the following structure:\n"
        "{\n"
        "  'decision': 'yes' or 'no',\n"
        "  'rationale': '<brief explanation>'\n"
        "}"
    ),
    "easyclinic_uc_id": (
        "You are an expert in software traceability. "
        "You are given two artifacts from a healthcare system. "
        "(1) is a use case and (2) is an interaction diagram. "
        "Does (2) directly realize (1)?\n\n"
        "Respond strictly in JSON format with the following structure:\n"
        "{\n"
        "  'decision': 'yes' or 'no',\n"
        "  'rationale': '<brief explanation>'\n"
        "}"
    ),
    "cchit": (
        "You are an expert in software traceability. "
        "You are given two artifacts from a healthcare system. "
        "(1) is a requirement and (2) is a regulation. "
        "Does (1) directly satisfy (2)?\n\n"
        "Respond strictly in JSON format with the following structure:\n"
        "{\n"
        "  'decision': 'yes' or 'no',\n"
        "  'rationale': '<brief explanation>'\n"
        "}"
    ),
    "cm1": (
        "You are an expert in software traceability. "
        "You are given two artifacts from an aerospace system. "
        "(1) is a high-level requirement and (2) is a design element. "
        "Does (2) directly fulfill (1)?\n\n"
        "Respond strictly in JSON format with the following structure:\n"
        "{\n"
        "  'decision': 'yes' or 'no',\n"
        "  'rationale': '<brief explanation>'\n"
        "}"
    ),
}


def get_system_role_for_dataset(dataset: str) -> str:
    """Get appropriate system role prompt for a dataset.

    Args:
        dataset: Dataset name (e.g., 'EasyClinic_UC_TC', 'CM1_NASA', 'CCHIT')

    Returns:
        System role prompt string appropriate for the dataset's task

    Raises:
        ValueError: If dataset not recognized
    """
    dataset_lower = dataset.lower()

    for key, role in SYSTEM_ROLES.items():
        if key in dataset_lower:
            return role

    raise ValueError(
        f"Unknown dataset: {dataset}. "
        f"Expected one containing: {', '.join(SYSTEM_ROLES.keys())}"
    )
