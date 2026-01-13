"""
model_comparison.py
-------------------

This module provides utilities for comparing machine‑learning models
within each economic regime. It supports:

1. Building a cross‑model comparison table summarizing:
   - RMSE
   - MAE
   - R²
   - RMSE confidence intervals

2. Pairwise statistical comparisons between models using:
   - Paired t‑tests (parametric)
   - Wilcoxon signed‑rank tests (non‑parametric)
   - Cohen’s d effect size

3. Running all pairwise comparisons for a given regime.

These tools are used in evaluation notebooks to determine whether
performance differences between models are statistically meaningful.
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
from itertools import combinations   # Generate all model pairs
from scipy.stats import ttest_rel, wilcoxon
# ttest_rel: paired t-test for dependent samples
# wilcoxon: non-parametric paired test for non-normal error distributions


# ============================================================
# 1. BUILD CROSS-MODEL PERFORMANCE TABLE
# ============================================================

def build_cross_model_comparison(regime_name, model_results):
    """
    Build a comparison table summarizing model performance metrics.

    Parameters
    ----------
    regime_name : str
        Name of the regime (e.g., "Regime A").
    model_results : dict
        Dictionary of model evaluation results, where each entry contains:
            {
                "test_metrics": {"RMSE": ..., "MAE": ..., "R2": ...},
                "error_ci": (lower_CI, upper_CI)
            }

    Returns
    -------
    pandas.DataFrame
        A sorted table containing:
        - Regime
        - Model name
        - RMSE
        - MAE
        - R²
        - RMSE_CI_Lower
        - RMSE_CI_Upper

    Notes
    -----
    - Sorting by RMSE highlights the best-performing models.
    - This table is typically used for reporting and visualization.
    """
    rows = []

    for model, res in model_results.items():
        rows.append({
            "Regime": regime_name,
            "Model": model,
            "RMSE": res["test_metrics"]["RMSE"],
            "MAE": res["test_metrics"]["MAE"],
            "R2": res["test_metrics"]["R2"],
            "RMSE_CI_Lower": res["error_ci"][0],
            "RMSE_CI_Upper": res["error_ci"][1]
        })

    return pd.DataFrame(rows).sort_values("RMSE")


# ============================================================
# 2. PAIRWISE MODEL COMPARISON (STATISTICAL TESTS)
# ============================================================

def compare_models(regime, m1, m2, errors_1, errors_2):
    """
    Compare two models using paired statistical tests.

    Tests performed:
    - Paired t-test (parametric)
    - Wilcoxon signed-rank test (non-parametric)
    - Cohen’s d (effect size)

    Parameters
    ----------
    regime : str
        Regime name (e.g., "Regime A").
    m1 : str
        Name of model 1.
    m2 : str
        Name of model 2.
    errors_1 : array-like
        Prediction errors for model 1 (y_pred - y_true).
    errors_2 : array-like
        Prediction errors for model 2.

    Returns
    -------
    dict
        {
            "Regime": ...,
            "Model 1": ...,
            "Model 2": ...,
            "Paired t-stat": ...,
            "Paired t-p": ...,
            "Wilcoxon W": ...,
            "Wilcoxon p": ...,
            "Cohen's d": ...
        }

    Notes
    -----
    - If both models produce identical errors, Wilcoxon cannot run.
    - Cohen’s d quantifies effect size:
        • 0.2 = small
        • 0.5 = medium
        • 0.8 = large
    """
    # Paired t-test
    t_stat, p_t = ttest_rel(errors_1, errors_2)

    # Difference vector
    diff = errors_1 - errors_2

    # Wilcoxon test (fails if all differences are zero)
    if np.allclose(diff, 0):
        w_stat, p_w = np.nan, 1.0
    else:
        w_stat, p_w = wilcoxon(errors_1, errors_2)

    # Cohen's d effect size
    cohens_d = diff.mean() / diff.std(ddof=1)

    return {
        "Regime": regime,
        "Model 1": m1,
        "Model 2": m2,
        "Paired t-stat": t_stat,
        "Paired t-p": p_t,
        "Wilcoxon W": w_stat,
        "Wilcoxon p": p_w,
        "Cohen's d": cohens_d
    }


# ============================================================
# 3. RUN ALL PAIRWISE MODEL COMPARISONS
# ============================================================

def run_pairwise_tests(regime_name, model_results):
    """
    Run pairwise statistical comparisons for all models in a regime.

    Parameters
    ----------
    regime_name : str
        Name of the regime.
    model_results : dict
        Dictionary of model evaluation results, where each entry contains:
            {
                "errors": array-like of prediction errors,
                ...
            }

    Returns
    -------
    pandas.DataFrame
        Table of pairwise comparisons including:
        - t-test results
        - Wilcoxon results
        - Cohen’s d
    """
    comparisons = []

    # Generate all unique model pairs
    for (m1, r1), (m2, r2) in combinations(model_results.items(), 2):
        comp = compare_models(
            regime=regime_name,
            m1=m1,
            m2=m2,
            errors_1=r1["errors"],
            errors_2=r2["errors"]
        )
        comparisons.append(comp)

    return pd.DataFrame(comparisons)
