"""
feature_filtering.py
--------------------

This module provides lightweight but essential feature‑filtering utilities
used during model preparation. These functions remove unstable, uninformative,
or unusable columns **before** model training.

Included functions:
- drop_high_missing_cols(): removes columns with excessive missingness.
- drop_constant_and_near_constant_cols(): removes columns with no or minimal variance.

These steps help:
- reduce noise,
- prevent model instability,
- improve generalization,
- reduce dimensionality,
- avoid issues in scaling, encoding, and SHAP analysis.

This module is imported by the data preparation pipeline.
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np      # Numerical operations
import pandas as pd     # DataFrame manipulation


# ============================================================
# 1. REMOVE HIGH‑MISSING COLUMNS
# ============================================================

def drop_high_missing_cols(X, threshold=0.60):
    """
    Remove columns with missingness above a specified threshold.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    threshold : float, default=0.65
        Columns with > threshold proportion of missing values are removed.

    Returns
    -------
    X_out : pandas.DataFrame
        DataFrame with high‑missing columns removed.
    drop_cols : list of str
        Names of columns that were dropped.

    Notes
    -----
    - This function is typically applied **only to training data**.
    - Validation/test sets must drop the same columns to maintain schema consistency.
    - High‑missing columns often cause instability in imputers and encoders.
    """
    # Compute missing fraction per column
    missing_frac = X.isna().mean()

    # Identify columns exceeding threshold
    drop_cols = missing_frac[missing_frac > threshold].index.tolist()

    print("Dropped (high missing):", drop_cols)

    # Return filtered DataFrame + list of removed columns
    return X.drop(columns=drop_cols), drop_cols


# ============================================================
# 2. REMOVE CONSTANT & NEAR‑CONSTANT COLUMNS
# ============================================================

def drop_constant_and_near_constant_cols(X, dominance_threshold=0.99):
    """
    Remove columns where a single value dominates the distribution.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    dominance_threshold : float, default=0.99
        If the most frequent value accounts for >= threshold proportion
        of all values, the column is removed.

    Returns
    -------
    X_out : pandas.DataFrame
        DataFrame with constant/near‑constant columns removed.
    drop_cols : list of str
        Names of columns that were dropped.

    Notes
    -----
    - Near‑constant columns provide almost no predictive signal.
    - They can cause issues in:
        • scaling,
        • encoding,
        • model convergence,
        • SHAP value stability.
    - Removing them improves model robustness and reduces dimensionality.
    """
    drop_cols = [
        col for col in X.columns
        if X[col].value_counts(normalize=True, dropna=False).iloc[0] >= dominance_threshold
    ]

    print("Dropped (near-constant):", drop_cols)

    return X.drop(columns=drop_cols), drop_cols

def drop_multicollinear_cols(X, corr_threshold=0.9999):
    """
    Remove columns that are perfectly or near‑perfectly correlated with others.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    corr_threshold : float, default=0.9999
        Absolute correlation above this threshold triggers removal.
        (Use < 1.0 to avoid floating‑point issues.)

    Returns
    -------
    X_out : pandas.DataFrame
        DataFrame with multicollinear columns removed.
    drop_cols : list of str
        Names of columns that were dropped.

    Notes
    -----
    - Perfect or near‑perfect correlations (|r| >= 0.9999) indicate
      deterministic relationships:
        • duplicated variables,
        • inverse‑coded variables,
        • recoded versions of the same question,
        • grid‑question expansions,
        • derived variables.
    - Keeping both causes:
        • unstable coefficients (even with Ridge),
        • unreliable SHAP values,
        • inflated dimensionality,
        • poor generalization.
    - We keep the *first* variable in each correlated group
      and drop the rest.
    """
    # Compute absolute correlation matrix
    corr = X.corr().abs()

    # Upper triangle mask (avoid duplicate pairs)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Columns to drop: any column with correlation >= threshold
    drop_cols = [
        col for col in upper.columns
        if any(upper[col] >= corr_threshold)
    ]

    print("Dropped (multicollinear):", drop_cols)

    return X.drop(columns=drop_cols), drop_cols

