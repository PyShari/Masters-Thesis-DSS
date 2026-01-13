"""
metrics.py
----------

This module provides evaluation metrics and diagnostic tools for regression
models used in the thesis. It includes:

- Standard regression metrics:
    • RMSE (Root Mean Squared Error)
    • MAE (Mean Absolute Error)
    • R² (Coefficient of Determination)

- Error distribution diagnostics:
    • mean, median, std of prediction errors
    • quartiles
    • extreme over/under‑prediction

- Bootstrap confidence intervals for RMSE:
    • non‑parametric bootstrap sampling
    • configurable number of bootstrap iterations

These utilities are imported by modeling notebooks and evaluation scripts.
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. STANDARD REGRESSION METRICS
# ============================================================

def rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error (RMSE).

    RMSE penalizes large errors more heavily than MAE and is one of the
    most commonly used metrics for regression tasks.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    float
        RMSE value.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """
    Compute Mean Absolute Error (MAE).

    MAE measures the average magnitude of errors without considering
    their direction. It is more robust to outliers than RMSE.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    float
        MAE value.
    """
    return mean_absolute_error(y_true, y_pred)


def r2(y_true, y_pred):
    """
    Compute R-squared (coefficient of determination).

    R² measures the proportion of variance in the target variable that
    is explained by the model. Values range from:
    - negative (worse than baseline),
    - 0 (no explanatory power),
    - up to 1 (perfect fit).

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    float
        R² score.
    """
    return r2_score(y_true, y_pred)


# ============================================================
# 2. ERROR DISTRIBUTION ANALYSIS
# ============================================================

def error_analysis(y_true, y_pred):
    """
    Compute detailed error distribution statistics.

    Useful for diagnosing:
    - systematic over/under‑prediction,
    - skewed error distributions,
    - model bias,
    - extreme prediction failures.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    dict
        {
            "mean_error": float,
            "median_error": float,
            "std_error": float,
            "q25_error": float,
            "q75_error": float,
            "max_overprediction": float,
            "max_underprediction": float
        }
    """
    e = y_pred - y_true  # prediction error vector

    return {
        "mean_error": np.mean(e),
        "median_error": np.median(e),
        "std_error": np.std(e),
        "q25_error": np.quantile(e, 0.25),
        "q75_error": np.quantile(e, 0.75),
        "max_overprediction": np.max(e),
        "max_underprediction": np.min(e)
    }


# ============================================================
# 3. BOOTSTRAP CONFIDENCE INTERVAL FOR RMSE
# ============================================================

def bootstrap_rmse_ci(y_true, y_pred, n_boot=300, alpha=0.05):
    """
    Compute a bootstrap confidence interval for RMSE.

    This uses non‑parametric bootstrap sampling:
    - repeatedly resample (with replacement) from the prediction pairs,
    - compute RMSE for each bootstrap sample,
    - take percentiles to form a confidence interval.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    n_boot : int, default=300
        Number of bootstrap samples.
    alpha : float, default=0.05
        Significance level (0.05 → 95% CI).

    Returns
    -------
    (lower, upper) : tuple of floats
        Lower and upper bounds of the RMSE confidence interval.
    """
    rmses = []
    n = len(y_true)

    for _ in range(n_boot):
        # Sample indices with replacement
        idx = np.random.choice(n, n, replace=True)
        rmses.append(np.sqrt(mean_squared_error(y_true[idx], y_pred[idx])))

    lower = np.percentile(rmses, 100 * alpha / 2)
    upper = np.percentile(rmses, 100 * (1 - alpha / 2))

    return lower, upper
