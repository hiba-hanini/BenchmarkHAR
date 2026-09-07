"""
imput : df_features_all : pd.DataFrame containing all features for all subjects
output : df_features_all : pd.DataFrame containing only the features that are not highly correlated
"""

from pathlib import Path
import pandas as pd
from src.dimension_reduction.correlation_filter import get_columns_to_keep

def run_correlation_filter(dataset_name, config):
    """Legacy CLI hook; selection is now train-only inside model CV folds."""
    print(
        "[run_correlation_filter] No global feature selection was performed. "
        "Correlation selection is fitted independently on each training fold."
    )
