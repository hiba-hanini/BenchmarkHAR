import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass(frozen=True)
class CorrelationSelector:
    """Train-fitted feature subset for deterministic correlation filtering."""
    feature_names: tuple[str, ...]
    selected_feature_names: tuple[str, ...]
    threshold: float = 0.95
    method: str = "pearson"
    n_fit_samples: int = 0


def get_columns_to_keep(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.95,
    method: str = "pearson") -> list[str]:

    corr_matrix = df[feature_cols].corr(method=method).abs()

    # upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    to_drop = [
        col for col in upper.columns if (upper[col] > threshold).any()
    ]

    cols_to_keep = [c for c in feature_cols if c not in to_drop]

    print(
        f"[correlation_filter] {len(feature_cols)} input features -> "
        f"{len(to_drop)} dropped (corr > {threshold}) -> "
        f"{len(cols_to_keep)} kept"
    )

    return cols_to_keep


def fit_correlation_selector(
    X_train: pd.DataFrame,
    feature_names: list[str] | None = None,
    threshold: float = 0.95,
    method: str = "pearson",
) -> CorrelationSelector:
    """Fit a correlation selector on training samples only."""
    if feature_names is None:
        feature_names = list(X_train.columns)
    missing = set(feature_names) - set(X_train.columns)
    if missing:
        raise ValueError(f"Training data is missing selector features: {sorted(missing)}")
    selected = get_columns_to_keep(X_train, feature_names, threshold, method)
    if not selected:
        raise ValueError("Correlation selection removed every feature.")
    return CorrelationSelector(
        tuple(feature_names), tuple(selected), threshold, method, len(X_train),
    )


def select_features_by_correlation(
    X_train: pd.DataFrame,
    feature_names: list[str] | None = None,
    threshold: float = 0.95,
    method: str = "pearson",
) -> list[str]:
    """Return the selected feature names using training data only."""
    return list(fit_correlation_selector(X_train, feature_names, threshold, method).selected_feature_names)


def transform_with_correlation_selector(
    X: pd.DataFrame, selector: CorrelationSelector,
) -> pd.DataFrame:
    """Apply a train-fitted selector without recalculating correlations."""
    missing = set(selector.selected_feature_names) - set(X.columns)
    if missing:
        raise ValueError(f"Cannot apply selector; data is missing: {sorted(missing)}")
    return X.loc[:, list(selector.selected_feature_names)].copy()
