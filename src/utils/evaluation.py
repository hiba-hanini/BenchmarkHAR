"""Standardized evaluation metric utilities for BenchmarkHAR.

Provides:
1. Fold-level Macro-F1 computed over classes present in the test fold ground truth,
   with `zero_division=0`. This measures classification discrimination on the
   activities the participant actually performed without penalizing unrecorded tasks.
2. Pooled Macro-F1 computed by concatenating ground truth and predictions across
   test folds. For Leave-One-Subject-Out (LOSO), test partitions are mutually disjoint,
   so pooled evaluation measures overall population performance over all protocol classes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_fold_metrics(
    y_true: Union[Sequence[int], np.ndarray],
    y_pred: Union[Sequence[int], np.ndarray],
    canonical_classes: Optional[Sequence[int]] = None,
    zero_division: int = 0,
) -> Dict[str, Any]:
    """Compute standardized classification metrics for a single evaluation fold.

    The primary `f1_macro` is computed over the classes present in the test fold's
    ground truth. When `canonical_classes` is provided, a complementary
    `f1_macro_canonical` is also recorded for comparison.

    Parameters
    ----------
    y_true : Sequence[int] or np.ndarray
        Ground-truth class identifiers for the test partition.
    y_pred : Sequence[int] or np.ndarray
        Predicted class identifiers for the test partition.
    canonical_classes : Sequence[int], optional
        The complete canonical class vocabulary defined by the dataset configuration.
    zero_division : int, default=0
        Value to return when a class has no positive predictions or ground-truth instances.

    Returns
    -------
    dict
        - 'f1_macro': Macro-averaged F1 over classes present in y_true.
        - 'accuracy': Overall classification accuracy.
        - 'present_classes': Sorted list of unique class IDs present in y_true.
        - 'per_class_f1': Dict mapping each present class ID to its individual F1 score.
        - 'f1_macro_canonical': (Optional) Macro-averaged F1 over canonical_classes.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_t)}) != y_pred ({len(y_p)})"
        )
    if len(y_t) == 0:
        raise ValueError("Cannot compute evaluation metrics on empty test partition.")

    # Unique classes present in the test fold ground truth
    present_classes = sorted(list(set(y_t.tolist())))

    # Primary metric: Macro-F1 over classes present in the test fold
    macro_f1 = float(
        f1_score(
            y_t,
            y_p,
            average="macro",
            labels=present_classes,
            zero_division=zero_division,
        )
    )

    acc = float(accuracy_score(y_t, y_p))

    # Per-class F1 for classes present in test ground truth
    per_class_scores = f1_score(
        y_t,
        y_p,
        average=None,
        labels=present_classes,
        zero_division=zero_division,
    )
    per_class_dict = {
        cls_id: float(score) for cls_id, score in zip(present_classes, per_class_scores)
    }

    metrics: Dict[str, Any] = {
        "f1_macro": macro_f1,
        "accuracy": acc,
        "present_classes": present_classes,
        "per_class_f1": per_class_dict,
    }

    # Optional complementary metric: fixed canonical vocabulary
    if canonical_classes is not None:
        canonical_classes_clean = [int(c) for c in canonical_classes]
        metrics["f1_macro_canonical"] = float(
            f1_score(
                y_t,
                y_p,
                average="macro",
                labels=canonical_classes_clean,
                zero_division=zero_division,
            )
        )

    return metrics


def compute_pooled_metrics(
    fold_y_true: Sequence[Sequence[int]],
    fold_y_pred: Sequence[Sequence[int]],
    zero_division: int = 0,
    cv_strategy: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute global pooled metrics by concatenating test samples across all folds.

    Notes on Split Overlap:
    - For Leave-One-Subject-Out (LOSO), test folds are mutually disjoint and their union
      covers every subject exactly once. In this setting, pooling is mathematically exact
      and provides a unified population evaluation across all canonical classes.
    - For GroupShuffleSplit (used on WISDM / Capture-24), folds are randomly sampled
      with replacement. Some subjects may appear in multiple test folds while others
      may be omitted, so pooling would over-weight repeatedly sampled participants.

    Parameters
    ----------
    fold_y_true : Sequence of Sequences
        Ground truth arrays for each test fold.
    fold_y_pred : Sequence of Sequences
        Prediction arrays for each test fold.
    zero_division : int, default=0
        Value for zero division.
    cv_strategy : str, optional
        Cross-validation strategy name (e.g. 'LOSO', 'GroupShuffleSplit').

    Returns
    -------
    dict
        - 'f1_macro_pooled': Macro-F1 over all classes present in the pooled ground truth.
        - 'accuracy_pooled': Accuracy on the concatenated test population.
        - 'pooled_present_classes': List of unique classes in the pooled ground truth.
        - 'n_pooled_samples': Total number of test samples pooled.
        - 'is_disjoint_partition': True if LOSO, False if GroupShuffleSplit, None if unspecified.
    """
    if len(fold_y_true) == 0 or len(fold_y_pred) == 0:
        raise ValueError("Cannot compute pooled metrics on empty fold lists.")

    pooled_true = np.concatenate([np.asarray(yt) for yt in fold_y_true])
    pooled_pred = np.concatenate([np.asarray(yp) for yp in fold_y_pred])

    if len(pooled_true) != len(pooled_pred):
        raise ValueError(
            f"Length mismatch: pooled_true ({len(pooled_true)}) != pooled_pred ({len(pooled_pred)})"
        )
    if len(pooled_true) == 0:
        raise ValueError("Cannot compute pooled metrics on empty samples.")

    pooled_present_classes = sorted(list(set(pooled_true.tolist())))

    pooled_macro_f1 = float(
        f1_score(
            pooled_true,
            pooled_pred,
            average="macro",
            labels=pooled_present_classes,
            zero_division=zero_division,
        )
    )

    pooled_acc = float(accuracy_score(pooled_true, pooled_pred))

    per_class_scores = f1_score(
        pooled_true,
        pooled_pred,
        average=None,
        labels=pooled_present_classes,
        zero_division=zero_division,
    )
    per_class_dict = {
        cls_id: float(score) for cls_id, score in zip(pooled_present_classes, per_class_scores)
    }

    is_disjoint = cv_strategy.upper() == "LOSO" if cv_strategy else None

    return {
        "f1_macro_pooled": pooled_macro_f1,
        "accuracy_pooled": pooled_acc,
        "pooled_present_classes": pooled_present_classes,
        "per_class_f1_pooled": per_class_dict,
        "n_pooled_samples": len(pooled_true),
        "is_disjoint_partition": is_disjoint,
    }
