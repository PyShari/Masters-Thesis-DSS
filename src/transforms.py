"""
transforms.py
-------------

This module provides transformation utilities used throughout the
modeling pipeline. It includes:

1. Target transformations:
   - log1p transform (with clipping)
   - inverse transform
   - Duan smearing correction for unbiased back‑transformation

2. LogTransformer:
   - A scikit‑learn compatible transformer that applies log transforms
     to selected numeric columns while safely handling zeros/negatives.

3. Feature filtering helpers (duplicated for convenience):
   - drop_high_missing_cols()
   - drop_constant_and_near_constant_cols()

4. detect_col_types():
   - Automatically classifies columns into:
       • binary
       • ordinal
       • continuous
       • categorical
     and identifies skewed numeric columns.

This module is imported by the data preparation and preprocessing pipeline.
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
# BaseEstimator + TransformerMixin allow LogTransformer to integrate
# seamlessly into scikit‑learn Pipelines.


# ============================================================
# 1. TARGET TRANSFORMATIONS
# ============================================================

def transform_target(y):
    """
    Apply log1p transform to the target variable.

    - Clips negative values to zero (log undefined for negatives).
    - Uses log1p(y) = log(1 + y), which is stable for small values.

    Parameters
    ----------
    y : array-like

    Returns
    -------
    array-like
        Log-transformed target.
    """
    return np.log1p(np.clip(y, 0, None))


def inverse_target(y):
    """
    Inverse of log1p transform.

    Parameters
    ----------
    y : array-like (log-transformed)

    Returns
    -------
    array-like
        Original scale target values.
    """
    return np.expm1(y)


def inverse_target_corrected(log_pred, log_residuals):
    """
    Apply Duan smearing correction for unbiased back-transformation.

    When a model is trained on log-transformed targets, simply applying
    exp() to predictions introduces bias. Duan's smearing estimator
    corrects this by multiplying by the mean of exp(residuals).

    Parameters
    ----------
    log_pred : array-like
        Predicted log-scale values.
    log_residuals : array-like
        Residuals on the log scale (y_log_true - y_log_pred).

    Returns
    -------
    array-like
        Bias-corrected predictions on the original scale.
    """
    smearing_factor = np.mean(np.exp(log_residuals))
    return np.expm1(log_pred) * smearing_factor


# ============================================================
# 2. LOG TRANSFORMER (SCIKIT-LEARN COMPATIBLE)
# ============================================================

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit‑learn compatible transformer that applies log transforms
    to selected numeric columns.

    Features:
    - Automatically avoids log(0) and negative values by replacing them
      with a small positive constant (1e‑6).
    - Handles infinite values by converting them to NaN.
    - Works seamlessly inside a Pipeline or ColumnTransformer.

    Parameters
    ----------
    columns : list of str or None
        Columns to transform. If None, all numeric columns are used.
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.cols_ = [
            c for c in (self.columns or X.columns)
            if c in X.columns and np.issubdtype(X[c].dtype, np.number)
        ]
        return self

    def transform(self, X):
        X = X.copy()

        for c in self.cols_:
            v = X[c]

            # Convert invalid strings to NaN
            v = pd.to_numeric(v, errors="coerce")

            # Replace inf with NaN
            v = v.replace([np.inf, -np.inf], np.nan)

            # Replace non-positive with small constant
            v = np.where(v <= 0, 1e-6, v)

            # Apply log1p for stability
            X[c] = pd.Series(np.log1p(v), index=X.index)

        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features if input_features is not None else self.cols_)


# ============================================================
# 3. FEATURE FILTERING HELPERS
# ============================================================

def drop_high_missing_cols(X, threshold=0.65):
    """
    Remove columns with missingness above a specified threshold.

    Parameters
    ----------
    X : pandas.DataFrame
    threshold : float, default=0.65

    Returns
    -------
    X_out : pandas.DataFrame
    drop_cols : list of str
    """
    missing_frac = X.isna().mean()
    drop_cols = missing_frac[missing_frac > threshold].index.tolist()

    print("Dropped (high missing):", drop_cols)
    return X.drop(columns=drop_cols), drop_cols


def drop_constant_and_near_constant_cols(X, dominance_threshold=0.99):
    """
    Remove columns where a single value dominates the distribution.

    Parameters
    ----------
    X : pandas.DataFrame
    dominance_threshold : float, default=0.99

    Returns
    -------
    X_out : pandas.DataFrame
    drop_cols : list of str
    """
    drop_cols = [
        col for col in X.columns
        if X[col].value_counts(normalize=True, dropna=False).iloc[0] >= dominance_threshold
    ]

    print("Dropped (near-constant):", drop_cols)
    return X.drop(columns=drop_cols), drop_cols


# ============================================================
# 4. COLUMN TYPE DETECTION
# ============================================================

def detect_col_types(df):
    """
    Automatically classify columns into:
    - binary (0/1)
    - ordinal (integer-valued)
    - continuous (float-valued)
    - categorical (non-numeric)

    Also identifies skewed continuous columns (|skew| > 1).

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    binary : list of str
    ordinal : list of str
    cont : list of str
    cat : list of str
    skewed_cols : list of str
        Continuous columns with high skewness.
    """
    binary, ordinal, cont, cat = [], [], [], []

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            u = pd.Series(df[c].dropna().unique())

            # Binary: exactly two values, both 0/1
            if len(u) == 2 and set(u).issubset({0, 1}):
                binary.append(c)

            # Ordinal: integer-valued numeric columns
            elif pd.api.types.is_integer_dtype(df[c]):
                ordinal.append(c)

            # Continuous: float-valued numeric columns
            else:
                cont.append(c)

        else:
            cat.append(c)

    # Identify skewed continuous columns
    skewed_cols = (
        df[cont].skew().abs().pipe(lambda s: s[s > 1].index.tolist())
        if cont else []
    )

    return binary, ordinal, cont, cat, skewed_cols
