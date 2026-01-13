"""
feature_importance.py
-----------------------

This module provides helper functions for extracting feature names and
model‑specific feature importance values from trained pipelines.

It supports:
- Ridge Regression coefficients,
- Random Forest Gini importances,
- XGBoost feature importances (gain, weight, cover, etc.).

These utilities are used in notebooks to interpret trained models and
generate ranked feature importance tables.
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd      # Tabular output for feature importance tables
import numpy as np       # Numerical operations


# ============================================================
# 1. FEATURE NAME EXTRACTION
# ============================================================

def get_feature_names(pipeline, X):
    """
    Extract feature names after preprocessing.

    Parameters
    ----------
    pipeline : sklearn.Pipeline
        The full modeling pipeline containing a 'preprocessor' step.
    X : pandas.DataFrame
        The dataset used to infer fallback column names if needed.

    Returns
    -------
    list of str
        Feature names after transformation.

    Notes
    -----
    - ColumnTransformer supports get_feature_names_out() in newer versions.
    - If unavailable (older sklearn), fallback to original X column names.
    """
    pre = pipeline.named_steps["preprocessor"]

    try:
        # Preferred: works when ColumnTransformer supports feature name export
        return pre.get_feature_names_out()
    except Exception:
        # Fallback: return original column names
        return X.columns.tolist()


# ============================================================
# 2. RIDGE REGRESSION COEFFICIENTS
# ============================================================

def ridge_coefficients(model_res):
    """
    Extract and return Ridge Regression coefficients.

    Parameters
    ----------
    model_res : dict
        Dictionary containing:
        - "pipeline": trained sklearn Pipeline
        - "X_test": test feature matrix

    Returns
    -------
    pandas.DataFrame
        Columns:
        - feature
        - coefficient
        Sorted by coefficient magnitude (descending).
    """
    pipeline = model_res["pipeline"]
    model = pipeline.named_steps["model"]
    X_test = model_res["X_test"]

    # Extract transformed feature names
    feature_names = get_feature_names(pipeline, X_test)

    # Ridge stores coefficients in model.coef_
    coefs = model.coef_

    return (
        pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        .sort_values("coefficient", ascending=False)
    )


# ============================================================
# 3. RANDOM FOREST GINI IMPORTANCE
# ============================================================

def rf_gini_importance(model_res):
    """
    Extract Random Forest Gini importances.

    Parameters
    ----------
    model_res : dict
        Dictionary containing:
        - "pipeline": trained sklearn Pipeline
        - "X_test": test feature matrix

    Returns
    -------
    pandas.DataFrame
        Columns:
        - feature
        - gini_importance
        Sorted by importance (descending).
    """
    pipeline = model_res["pipeline"]
    model = pipeline.named_steps["model"]
    X_test = model_res["X_test"]

    feature_names = get_feature_names(pipeline, X_test)
    importances = model.feature_importances_

    return (
        pd.DataFrame({"feature": feature_names, "gini_importance": importances})
        .sort_values("gini_importance", ascending=False)
    )


# ============================================================
# 4. XGBOOST FEATURE IMPORTANCE
# ============================================================

def xgb_importance(model_res, importance_type="gain"):
    """
    Extract XGBoost feature importances.

    Parameters
    ----------
    model_res : dict
        Dictionary containing:
        - "pipeline": trained sklearn Pipeline
        - "X_test": test feature matrix
    importance_type : str, default="gain"
        XGBoost importance type:
        - "gain": average gain of splits using the feature
        - "weight": number of times a feature is used in splits
        - "cover": number of samples affected by splits

    Returns
    -------
    pandas.DataFrame
        Columns:
        - feature
        - importance
        Sorted by importance (descending).

    Notes
    -----
    - XGBoost returns feature names like "f0", "f1", ...
    - These must be mapped back to transformed feature names.
    """
    pipeline = model_res["pipeline"]
    model = pipeline.named_steps["model"]
    X_test = model_res["X_test"]

    # Extract transformed feature names
    feature_names = get_feature_names(pipeline, X_test)

    # Raw importance dict: {"f0": value, "f12": value, ...}
    raw = model.get_booster().get_score(importance_type=importance_type)

    # Map "f0" → feature_names[0], etc.
    mapped = {
        feature_names[int(k[1:])]: v
        for k, v in raw.items()
    }

    return (
        pd.DataFrame({"feature": list(mapped.keys()), "importance": list(mapped.values())})
        .sort_values("importance", ascending=False)
    )
