"""
modeling.py
-----------

This module defines the full modeling and evaluation framework used across all regimes.

It includes:

PART 1 — Unified Modeling Framework
1. evaluate_model() — unified evaluation for all models
2. run_full_regime_analysis() — full experiment runner
3. run_median_baseline() — baseline model
4. run_ridge() — Ridge regression model

PART 2 — Tree-Based Models & Full Experiment Runner
5. run_rf() — Random Forest training + evaluation
6. run_xgb() — XGBoost training + evaluation
7. run_all_models_for_regime() — full experiment runner for a regime

All models follow the same pipeline structure:
    preprocessor → model → evaluation → SHAP compatibility

This ensures:
- consistent metrics,
- reproducible comparisons across regimes,
- seamless integration with SHAP interpretability tools.
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.dummy import DummyRegressor

from src.data_preparation import prepare_data
from src.feature_engineering import build_preprocessor
from src.transforms import (
    detect_col_types,
    inverse_target_corrected,
    inverse_target,
    transform_target
)
from src.metrics import rmse, mae, r2, error_analysis, bootstrap_rmse_ci
from src.feature_importance import (
    ridge_coefficients,
    rf_gini_importance,
    xgb_importance
)
from src.model_comparison import run_pairwise_tests, build_cross_model_comparison
from src.plotting import (
    plot_error_distribution,
    plot_per_fold_rmse,
    plot_predicted_vs_actual
)


# ============================================================
# 1. UNIFIED MODEL EVALUATION
# ============================================================
from sklearn.model_selection import PredefinedSplit

def evaluate_model(
    pipeline,
    search,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    config
):
    """
    Evaluate a regression model using a unified framework.
    
    This function supports:
    - Baseline models (no hyperparameter search)
    - Models trained with RandomizedSearchCV using a predefined split
    - Smearing correction for log-transformed targets
    - Extraction of model-specific feature importance metrics
    
    Returns a structured dictionary containing:
    - Predictions
    - Errors
    - Metrics
    - Confidence intervals
    - Feature names
    - Model object and CV results
    """

    # =====================================================
    # SPECIAL CASE: Baseline model (no hyperparameter search)
    # =====================================================
    if search is None:
        # Fit the pipeline directly on training data
        pipeline.fit(X_train, y_train)

        # Predict on test set (no log transform or smearing)
        y_test_pred = pipeline.predict(X_test)
        y_test_true = y_test

        # Compute evaluation metrics
        test_metrics = {
            "RMSE": rmse(y_test_true, y_test_pred),
            "MAE": mae(y_test_true, y_test_pred),
            "R2": r2(y_test_true, y_test_pred)
        }

        # Return baseline results
        return {
            "pipeline": pipeline,
            "best_params_overall": None,
            "cv_results": None,
            "val_rmse": None,
            "y_test_pred": y_test_pred,
            "y_test_true": y_test_true,
            "errors": y_test_pred - y_test_true,
            "test_metrics": test_metrics,
            "error_ci": bootstrap_rmse_ci(y_test_true.values, y_test_pred),
            "error_analysis": error_analysis(y_test_true, y_test_pred),
            "feature_names": None,
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test
        }

    # =====================================================
    # Combine TRAIN + VAL for RandomizedSearchCV
    # =====================================================
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    # =====================================================
    # Fit hyperparameter search
    # =====================================================
    search.fit(X_train_val, y_train_val)
    best_model = search.best_estimator_

    # =====================================================
    # Smearing correction for unbiased back-transformation
    # =====================================================
    if config["log_target"]:
        # predictions in log space
        log_pred_train = best_model.predict(X_train)
        log_residuals = y_train - log_pred_train
    
        log_pred_test = best_model.predict(X_test)
        y_test_pred = inverse_target_corrected(log_pred_test, log_residuals)
    
        y_test_true = np.expm1(y_test)
        errors = y_test_pred - y_test_true
    
    else:
        # predictions already in real space
        y_test_pred = best_model.predict(X_test)
        y_test_true = y_test
        errors = y_test_pred - y_test_true

    # =====================================================
    # Compute evaluation metrics
    # =====================================================
    test_metrics = {
        "RMSE": rmse(y_test_true, y_test_pred),
        "MAE": mae(y_test_true, y_test_pred),
        "R2": r2(y_test_true, y_test_pred)
    }

    error_ci = bootstrap_rmse_ci(y_test_true.values, y_test_pred)
    err_analysis = error_analysis(y_test_true, y_test_pred)

    # =====================================================
    # Extract validation RMSE from CV results
    # =====================================================
    best_idx = search.best_index_
    val_rmse = -search.cv_results_["mean_test_score"][best_idx]

    # =====================================================
    # Extract feature names
    # =====================================================
    preprocessor = best_model.named_steps["preprocessor"]
    raw_feature_names = preprocessor.get_feature_names_out()

    fixed_feature_names = []
    for name in raw_feature_names:
        if name.startswith("cat_high_"):
            idx = int(name.split("_")[-1])
            original_col = preprocessor.transformers_[4][2][idx]
            fixed_feature_names.append(f"{original_col}_target")
        else:
            fixed_feature_names.append(name)

    # =====================================================
    # Build result dictionary
    # =====================================================
    res = {
        "pipeline": best_model,
        "best_params_overall": search.best_params_,
        "cv_results": search.cv_results_,
        "val_rmse": val_rmse,
        "y_test_pred": y_test_pred,
        "y_test_true": y_test_true,
        "errors": errors,
        "test_metrics": test_metrics,
        "error_ci": error_ci,
        "error_analysis": err_analysis,
        "feature_names": fixed_feature_names,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test, 
        "kept_features": X_train.columns.tolist()
    }

    # =====================================================
    # Model-specific feature importance extraction
    # =====================================================
    model_name = type(best_model.named_steps["model"]).__name__

    if model_name == "Ridge":
        res["ridge_coefficients"] = ridge_coefficients(res)

    if model_name == "RandomForestRegressor":
        res["gini_importance"] = rf_gini_importance(res)

    if model_name == "XGBRegressor":
        res["xgb_gain"] = xgb_importance(res, "gain")
        res["xgb_weight"] = xgb_importance(res, "weight")
        res["xgb_cover"] = xgb_importance(res, "cover")

    return res


def run_median_baseline(df, config):
    """
    Train and evaluate a simple baseline model that predicts the median
    of the target variable. This provides a lower bound for model performance.
    """

    # -----------------------------
    # Data preparation
    # -----------------------------
    X_train, X_val, X_test, y_train, y_val, y_test, dropped = prepare_data(df, config)
    binary, ordinal, cont, cat, skewed_cols = detect_col_types(X_train)

    # -----------------------------
    # Preprocessor
    # -----------------------------
    preprocessor = build_preprocessor(X_train, skewed_cols)

    # -----------------------------
    # Pipeline (no learning, just median prediction)
    # -----------------------------
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DummyRegressor(strategy="median"))
    ])

    # No hyperparameter tuning for baseline
    search = None

    # -----------------------------
    # Evaluate using unified evaluation framework
    # -----------------------------
    res = evaluate_model(
        pipeline, search,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        config
    )

    # -----------------------------
    # Final output
    # -----------------------------
    res.update({
        "dropped_features": dropped,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    })

    return res

def run_ridge(df, config):
    """
    Train and evaluate a Ridge regression model with hyperparameter tuning.
    Ridge is a linear model with L2 regularization, useful for stability
    when predictors are correlated.
    """

    # -----------------------------
    # Data preparation
    # -----------------------------
    X_train, X_val, X_test, y_train, y_val, y_test, dropped = prepare_data(df, config)
    binary, ordinal, cont, cat, skewed_cols = detect_col_types(X_train)

    # -----------------------------
    # Preprocessor
    # -----------------------------
    preprocessor = build_preprocessor(X_train, skewed_cols)

    # -----------------------------
    # Pipeline
    # -----------------------------
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge())
    ])

    # -----------------------------
    # Hyperparameter Search
    # -----------------------------
    search = RandomizedSearchCV(
        pipeline,
        param_distributions={
            "model__alpha": [10, 100, 1000]
        },
        n_iter=8,
        cv=config["cv_splits"],
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=config["random_state"]
    )

    # -----------------------------
    # Evaluate
    # -----------------------------
    res = evaluate_model(
        pipeline, search,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        config
    )

    # -----------------------------
    # Final output
    # -----------------------------
    res.update({
        "dropped_features": dropped,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    })

    return res

# ============================================================
# 1. RANDOM FOREST REGRESSOR
# ============================================================

def run_rf(df, config):
    """
    Train and evaluate a Random Forest regressor.

    Random Forests are robust, non-linear models that:
    - handle mixed data types,
    - capture interactions automatically,
    - are resistant to overfitting with enough trees.

    Parameters
    ----------
    df : pandas.DataFrame
        Regime dataset.
    config : dict
        GLOBAL_CONFIG dictionary.

    Returns
    -------
    dict
        Full evaluation output from evaluate_model().
    """

    # -----------------------------
    # Data preparation
    # -----------------------------
    X_train, X_val, X_test, y_train, y_val, y_test, dropped = prepare_data(df, config)
    binary, ordinal, cont, cat, skewed_cols = detect_col_types(X_train)

    # -----------------------------
    # Preprocessor
    # -----------------------------
    preprocessor = build_preprocessor(X_train, skewed_cols)

    # -----------------------------
    # Pipeline
    # -----------------------------
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            random_state=config["random_state"],
            n_jobs=-1
        ))
    ])

    # -----------------------------
    # Hyperparameter Search
    # -----------------------------
    search = RandomizedSearchCV(
        pipeline,
        param_distributions={
            "model__n_estimators": [300, 500],
            "model__max_depth": [8, 10, 12, 14],
            "model__min_samples_leaf": [1, 2, 3, 5],
            "model__max_features": ["sqrt", "log2", 0.5, 0.7]
        },
        n_iter=20,
        cv=config["cv_splits"],
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=config["random_state"]
    )

    # -----------------------------
    # Evaluate
    # -----------------------------
    res = evaluate_model(
        pipeline, search,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        config
    )

    # -----------------------------
    # Final output
    # -----------------------------
    res.update({
        "dropped_features": dropped,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    })

    return res


# ============================================================
# 2. XGBOOST REGRESSOR
# ============================================================

from scipy.stats import loguniform

def run_xgb(df, config):
    """
    Train and evaluate an XGBoost regressor.

    XGBoost is a high‑performance gradient boosting algorithm that:
    - handles non-linearities,
    - captures complex interactions,
    - performs exceptionally well on structured/tabular data.

    Parameters
    ----------
    df : pandas.DataFrame
        Regime dataset.
    config : dict
        GLOBAL_CONFIG dictionary.

    Returns
    -------
    dict
        Full evaluation output from evaluate_model().
    """

    # -----------------------------
    # Data preparation
    # -----------------------------
    X_train, X_val, X_test, y_train, y_val, y_test, dropped = prepare_data(df, config)
    binary, ordinal, cont, cat, skewed_cols = detect_col_types(X_train)

    # -----------------------------
    # Preprocessor
    # -----------------------------
    preprocessor = build_preprocessor(X_train, skewed_cols)

    # -----------------------------
    # Pipeline
    # -----------------------------
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=config["random_state"],
            n_jobs=-1,
            n_estimators=500,
            eval_metric="rmse"
        ))
    ])

    # -----------------------------
    # Hyperparameter Search
    # -----------------------------
    search = RandomizedSearchCV(
        pipeline,
        param_distributions={
            "model__max_depth": [3, 4,5, 6],
            "model__min_child_weight": [1, 3, 5, 7],
            "model__gamma": [0.005, 0.009, 0.01],
            "model__subsample": [0.6, 0.77, 0.8],
            "model__colsample_bytree": [0.7, 0.75, 0.8],
            "model__reg_alpha": [0.001, 0.005, 0.1],
            "model__reg_lambda": [0.03, 0.04, 0.05],
            "model__learning_rate": [0.02, 0.03, 0.05]
        },
        n_iter=40,
        cv=config["cv_splits"],
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=config["random_state"]
    )

    # -----------------------------
    # Evaluate
    # -----------------------------
    res = evaluate_model(
        pipeline, search,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        config
    )

    # -----------------------------
    # Final output
    # -----------------------------
    res.update({
        "dropped_features": dropped,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    })

    return res


# ============================================================
# 3. RUN ALL MODELS FOR A REGIME
# ============================================================

def run_all_models_for_regime(regime_name, df, config):
    """
    Run all models (baseline + Ridge + RF + XGB) for a given regime.

    Produces:
    - Model results
    - Cross‑model comparison table
    - Pairwise statistical tests
    - Diagnostic plots (errors, RMSE curves, predicted vs actual)

    Parameters
    ----------
    regime_name : str
        Name of the regime (e.g., "Regime A").
    df : pandas.DataFrame
        Regime dataset.
    config : dict
        GLOBAL_CONFIG dictionary.

    Returns
    -------
    dict
        {
            "results": {...},
            "comparison_table": DataFrame,
            "stats_table": DataFrame
        }
    """

    print(f"\n{'='*80}")
    print(f"RUNNING FULL EXPERIMENT – {regime_name}")
    print(f"{'='*80}")

    # ------------------
    # Train all models
    # ------------------
    results = {
        "MedianBaseline": run_median_baseline(df, config),
        "Ridge": run_ridge(df, config),
        "RandomForest": run_rf(df, config),
        "XGBoost": run_xgb(df, config)
    }

    # ------------------
    # Comparison Tables
    # ------------------
    comparison_table = build_cross_model_comparison(regime_name, results)
    stats_table = run_pairwise_tests(regime_name, results)

    # ------------------
    # Diagnostic Plots
    # ------------------
    plot_error_distribution(results, regime_name)
    plot_per_fold_rmse(results, regime_name)

    for model, res in results.items():
        plot_predicted_vs_actual(
            model,
            res["y_test_true"],
            res["y_test_pred"],
            regime_name
        )

    return {
        "results": results,
        "comparison_table": comparison_table,
        "stats_table": stats_table
    }
