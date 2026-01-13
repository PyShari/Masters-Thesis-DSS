"""
target_filtering.py
-------------------

This module provides utilities for filtering out rows with missing
target values. It is used during the early stages of preprocessing
and again after splitting the dataset into economic regimes.

Functions included:
- drop_missing_target(): remove rows where the target is missing.
- drop_missing_target_for_regimes(): apply the same filtering to
  each regime-specific DataFrame.

These steps ensure that all downstream modeling receives clean,
valid target values.
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd      # DataFrame manipulation
import numpy as np       # Numerical operations (not strictly required here but kept for consistency)


# ============================================================
# 1. DROP ROWS WITH MISSING TARGET VALUES
# ============================================================

def drop_missing_target(df, target="127"):
    """
    Remove rows where the target variable is missing.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    target : str, default="127"
        Name of the target column to check for missing values.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame containing only rows where the target is present.

    Notes
    -----
    - This function is non-destructive (returns a copy).
    - Column names are normalized to string to avoid dtype mismatches.
    - Raises a ValueError if the target column is not found.
    """
    df = df.copy()

    # Ensure column names are strings (LISS sometimes mixes int/str)
    df.columns = df.columns.astype(str)

    # Validate target column
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    # Filter rows where target is not missing
    df_out = df[df[target].notna()].copy()

    # Diagnostics
    removed = df.shape[0] - df_out.shape[0]
    print(f"Rows removed due to missing target '{target}': {removed}")
    print(f"Remaining rows after filtering: {df_out.shape[0]}")

    return df_out


# ============================================================
# 2. APPLY TARGET FILTERING TO ALL REGIMES
# ============================================================

def drop_missing_target_for_regimes(regimes, target="127"):
    """
    Apply drop_missing_target() to all regime DataFrames.

    Parameters
    ----------
    regimes : dict
        Dictionary of regime DataFrames, e.g.:
            {
                "A": df_regime_a,
                "B": df_regime_b,
                "C": df_regime_c
            }
    target : str
        Target column name.

    Returns
    -------
    dict
        Updated dictionary with filtered DataFrames.

    Notes
    -----
    - Each regime is processed independently.
    - Ensures that all regime subsets contain valid target values.
    """
    return {
        regime_name: drop_missing_target(df, target=target)
        for regime_name, df in regimes.items()
    }
