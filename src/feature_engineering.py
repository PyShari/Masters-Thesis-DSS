"""
feature_engineering.py
----------------------

This module defines the preprocessing pipeline used across all models.
It automatically detects column types, applies appropriate transformations,
and returns a scikit‑learn ColumnTransformer ready for integration into
a full modeling pipeline.

Included functionality:
- Automatic detection of:
    • binary variables
    • ordinal integer variables
    • continuous numeric variables
    • categorical variables (low- and high-cardinality)
- Log-transform of skewed numeric variables
- Median imputation + missing indicators for numeric features
- One-hot encoding for low-cardinality categoricals
- Target encoding for high-cardinality categoricals
- Standard scaling for continuous variables

This preprocessing design ensures:
- consistent handling of missingness,
- robust encoding of categorical variables,
- improved model stability,
- compatibility with SHAP and downstream interpretability tools.
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from category_encoders import TargetEncoder

# Custom utilities
from src.transforms import detect_col_types, LogTransformer


# ============================================================
# 1. BUILD PREPROCESSOR
# ============================================================

def build_preprocessor(X_train, skewed_cols):
    """
    Build a full preprocessing pipeline for mixed-type survey data.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix used to infer column types.
    skewed_cols : list of str
        List of numeric columns identified as skewed (|skew| > 1)
        to be log-transformed.

    Returns
    -------
    ColumnTransformer
        A fitted preprocessing transformer ready to be used inside
        a scikit‑learn Pipeline.

    Notes
    -----
    - Column types are automatically detected via detect_col_types().
    - Categorical variables are split into:
        • low-cardinality (<= 15 unique values) → OneHotEncoder
        • high-cardinality (> 15 unique values) → TargetEncoder
    - Numeric variables receive:
        • log-transform (optional)
        • median imputation + missing indicator
        • standard scaling
    """
    # ---------------------------------------------------------
    # Detect column types
    # ---------------------------------------------------------
    binary, ordinal, cont, cat, skewed_cols = detect_col_types(X_train)

    # Split categorical variables by cardinality
    cat_high = [c for c in cat if X_train[c].nunique() > 15]
    cat_low  = [c for c in cat if X_train[c].nunique() <= 15]

    # ---------------------------------------------------------
    # Numeric pipeline
    # ---------------------------------------------------------
    num_pipe = Pipeline([
        ("log", LogTransformer(columns=skewed_cols)),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler())
    ])

    # ---------------------------------------------------------
    # Low-cardinality categorical pipeline
    # ---------------------------------------------------------
    low_cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # ---------------------------------------------------------
    # High-cardinality categorical pipeline
    # ---------------------------------------------------------
    high_cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("target", TargetEncoder(smoothing=5.0))
    ])

    # ---------------------------------------------------------
    # Combine all pipelines into a ColumnTransformer
    # ---------------------------------------------------------
    preprocessor = ColumnTransformer([
        ("num", num_pipe, cont),
        ("binary", SimpleImputer(strategy="most_frequent"), binary),
        ("ordinal", SimpleImputer(strategy="most_frequent"), ordinal),
        ("cat_low", low_cat_pipe, cat_low),
        ("cat_high", high_cat_pipe, cat_high)
    ])

    return preprocessor
