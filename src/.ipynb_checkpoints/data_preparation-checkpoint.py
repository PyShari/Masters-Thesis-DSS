"""
data_preparation.py
-------------------

This module contains utilities for preparing data for machine‑learning
models. It includes:

1. prepare_data()
   - filters invalid target values,
   - applies optional log‑transformation,
   - performs train/validation/test split,
   - removes high‑missing and near‑constant features (train‑only),
   - ensures consistent feature sets across splits.

2. build_preprocessor()
   - automatically detects column types (binary, ordinal, continuous, categorical),
   - builds a ColumnTransformer with:
       • log transform for skewed numeric features,
       • median imputation + scaling for numeric features,
       • one‑hot encoding for low‑cardinality categorical features,
       • target encoding for high‑cardinality categorical features,
       • simple imputation for binary/ordinal features.

This module is imported by modeling notebooks and training scripts.
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd

# --- Scikit‑learn utilities ---
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# --- High‑cardinality categorical encoding ---
from category_encoders import TargetEncoder

# --- Custom feature filtering utilities ---
from src.feature_filtering import (
    drop_high_missing_cols,                 # Removes columns with excessive missingness
    drop_constant_and_near_constant_cols , 
    drop_multicollinear_cols 
)

# --- Custom transformers ---
from src.transforms import detect_col_types, LogTransformer
# detect_col_types: automatically classifies columns into binary/ordinal/continuous/categorical
# LogTransformer: applies log1p transform to selected numeric columns


# ============================================================
# 1. TRAIN/VAL/TEST SPLITTING + TARGET CLEANING
# ============================================================

def prepare_data(df, config):
    """
    Prepare dataset for modeling by applying:
    - target filtering,
    - optional log transformation,
    - train/validation/test split,
    - removal of high‑missing and near‑constant columns (train‑only).

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataset containing features + target.
    config : dict
        Global configuration dictionary (GLOBAL_CONFIG).

    Returns
    -------
    X_train, X_val, X_test : pandas.DataFrame
        Feature matrices for each split.
    y_train, y_val, y_test : pandas.Series
        Target vectors for each split.
    dropped_features : dict
        Lists of removed columns:
            {
                "high_missing": [...],
                "near_constant": [...]
            }

    Notes
    -----
    - Target filtering removes unrealistic weekly hours (<=0 or >60).
    - High‑missing and near‑constant columns are removed **only from training**
      and then dropped from validation/test to avoid leakage.
    """
    df = df.replace(r"^\s*$", np.nan, regex=True)

    y_col = config["y_col"]
    test_size = config["test_size"]
    val_size = config["val_size"]
    random_state = config["random_state"]
    log_target = config["log_target"]

    # 1. Filter invalid target values
    mask = (df[y_col] > 0) & (df[y_col] <= 60)
    df_filtered = df.loc[mask].copy()

    X = df_filtered.drop(columns=[y_col])
    X = X.drop(columns=["year"], errors="ignore")
    y = df_filtered[y_col].astype(float)

    # 2. Train/test split
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3. Validation split
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_ratio, random_state=random_state
    )

    dropped_features = {}

    # 4. Remove high-missing columns
    X_train, high_missing_cols = drop_high_missing_cols(X_train)
    X_val  = X_val.drop(columns=high_missing_cols, errors="ignore")
    X_test = X_test.drop(columns=high_missing_cols, errors="ignore")
    dropped_features["high_missing"] = high_missing_cols

    # 5. Remove constant / near-constant columns
    X_train, near_constant_cols = drop_constant_and_near_constant_cols(X_train)
    X_val  = X_val.drop(columns=near_constant_cols, errors="ignore")
    X_test = X_test.drop(columns=near_constant_cols, errors="ignore")
    dropped_features["near_constant"] = near_constant_cols

    # ---------------------------------------------------------
    # 6. Remove multicollinear columns (train-only)
    # ---------------------------------------------------------
    X_train, multicollinear_cols = drop_multicollinear_cols(X_train)
    X_val  = X_val.drop(columns=multicollinear_cols, errors="ignore")
    X_test = X_test.drop(columns=multicollinear_cols, errors="ignore")
    dropped_features["multicollinear"] = multicollinear_cols

    return X_train, X_val, X_test, y_train, y_val, y_test, dropped_features


# ============================================================
# 2. BUILD PREPROCESSOR (COLUMNTRANSFORMER)
# ============================================================

def build_preprocessor(X_train, skewed_cols):
    """
    Build a ColumnTransformer that apreprocesses numeric, categorical,
    binary, and ordinal features differently.

    Steps:
    - detect column types using detect_col_types()
    - apply log transform to skewed numeric features
    - impute + scale numeric features
    - impute + one‑hot encode low‑cardinality categorical features
    - impute + target encode high‑cardinality categorical features
    - impute binary and ordinal features with most frequent value

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix.
    skewed_cols : list of str
        Numeric columns that should receive log transformation.

    Returns
    -------
    preprocessor : sklearn.compose.ColumnTransformer
        A fully configured preprocessing pipeline.
    """
    # Automatically detect column types
    binary, ordinal, cont, cat, skewed_cols = detect_col_types(X_train)

    # Split categorical features by cardinality
    cat_high = [c for c in cat if X_train[c].nunique() > 15]   # high-cardinality → target encoding
    cat_low  = [c for c in cat if X_train[c].nunique() <= 15]  # low-cardinality → one-hot encoding

    # ---------------------------------------------------------
    # Numeric pipeline
    # ---------------------------------------------------------
    num_pipe = Pipeline([
        ("log", LogTransformer(columns=skewed_cols)),  # log1p transform for skewed features
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
        ("target", TargetEncoder(smoothing=5.0))  # reduces overfitting
    ])

    # ---------------------------------------------------------
    # Combine everything into a ColumnTransformer
    # ---------------------------------------------------------
    preprocessor = ColumnTransformer([
        ("num", num_pipe, cont),
        ("binary", SimpleImputer(strategy="most_frequent"), binary),
        ("ordinal", SimpleImputer(strategy="most_frequent"), ordinal),
        ("cat_low", low_cat_pipe, cat_low),
        ("cat_high", high_cat_pipe, cat_high)
    ])

    return preprocessor
