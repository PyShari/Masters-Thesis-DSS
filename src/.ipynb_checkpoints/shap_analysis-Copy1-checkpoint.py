"""
shap_analysis.py
----------------

This module provides utilities for computing and visualizing SHAP values
for different model families (linear, tree-based) within the thesis
pipeline. It includes:

1. Building a shared SHAP background dataset using KMeans clustering.
2. Computing SHAP values for:
       • Linear models (Ridge)
       • Tree-based models (Random Forest, XGBoost)
3. Generating SHAP summary plots.
4. Selecting the best-performing model for a regime.

These tools are used in the interpretability notebooks to analyze
feature contributions and compare model behavior across economic regimes.
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import shap
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor

# SHAP visualization utilities
from shap import summary_plot, dependence_plot

# Project modules
from src.feature_engineering import build_preprocessor
from src.transforms import detect_col_types
from src.modeling import evaluate_model
from src.feature_importance import (
    ridge_coefficients,
    rf_gini_importance,
    xgb_importance
)
from src.utils import safe_filename

# Statistical helpers
from scipy.stats import (
    spearmanr,
    randint,
    uniform,
    loguniform,
    skew,
    kurtosis
)

feature_name_map = {
    "num__143": "Has sideline job",
    "num__409": "Supervises employees",
    "num__446": "Works less to care for children",
    "binary__391": "Reason <36h: family situation",
    "binary__390": "Reason <36h: no opportunity to work more",
    "binary__393": "Reason <36h: home activities",
    "binary__401": "Reason <36h: other",
    "binary__400": "Reason <36h: more leisure",
    "binary__088": "Currently employed",
    "num__404": "Occupation (ISCO)",
    "num__402": "Sector",
    "num__122": "Retirement feeling",
    "num__289": "Expected retirement age",
    "num__008": "Education completed (2nd degree)",
    "num__479": "Savings deposit (2007)",
    "num__384": "Max full-time wage estimate",
    "num__136": "Commute to work (minutes)",
    "num__032": "Education–work match scale",
    "num__003": "Age",

    # NEW VARIABLES YOU LISTED
    "num__022": "Preferred retirement age",
    "num__459": "Lifecourse deposit (2007)",
    "num__438": "No children or grandchildren",
    "num__450": "Provides informal care",
    "num__123": "Organisation type (first job)",
    "num__035": "Has taken job courses",
    "binary__437": "Has grandchildren",
    "binary__492": "Reason <36h: family/health",

    # Missing indicators
    "num__missingindicator_141": "Missing: weekend work frequency",
    "num__missingindicator_142": "Missing: evening work frequency",
    "num__missingindicator_517": "Missing: minimum wage offer",
    "num__missingindicator_405": "Missing: occupation (first job)",
    "num__missingindicator": "Missing: irregular work hours",  # if this is the real column name

    # Existing ones
    "num__411": "Option to continue working after pension",
    "num__missingindicator_411": "Missing: option to continue after pension",
    "num__missingindicator_134": "Missing: year started working",
    "num__517": "Minimum wage offer (EUR)",
    "num__missingindicator_035": "Missing: took job courses",
    "num__missingindicator_528": "Missing: workplace size",
    "num__611": "Profession (ISCO 085)",
    "num__missingindicator_309": "Missing: desired part-time hours",
    "num__missingindicator_136": "Missing: commute to work (minutes)",
    "num__005": "Highest Dutch education",
    "num__308": "Retirement transition preference",
    "num__610": "Weekly home-work hours",
    "num__309": "Desired part-time hours",
    "num__134": "Year started working",
    "num__528": "Workplace size",
    "num__141": "Weekend work frequency"
}

# ============================================================
# 1. BUILD SHAP BACKGROUND DATASET
# ============================================================

def build_shared_shap_background(pipeline, X_train, n_clusters=100):
    """
    Build a compressed SHAP background dataset using KMeans clustering.

    SHAP explainers require a "background" dataset representing the
    distribution of the training data. Using the full training set is
    computationally expensive, so we compress it using KMeans cluster
    centers in the transformed feature space.

    Parameters
    ----------
    pipeline : sklearn.Pipeline
        Trained pipeline containing a 'preprocessor' step.
    X_train : pandas.DataFrame
        Training feature matrix.
    n_clusters : int, default=100
        Number of KMeans clusters to use for compression.

    Returns
    -------
    numpy.ndarray
        Array of cluster centers representing the SHAP background dataset.
    """
    # Transform training data into model-ready feature space
    X_trans = pipeline.named_steps["preprocessor"].transform(X_train)

    # Cluster the transformed data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_trans)

    # Use cluster centers as SHAP background
    return kmeans.cluster_centers_


# ============================================================
# 2. SHAP FOR LINEAR MODELS
# ============================================================

def shap_linear(pipeline, X_train, X_test, background):
    """
    Compute SHAP values for linear models (e.g., Ridge Regression).

    Uses SHAP's LinearExplainer, which is optimized for linear models
    and supports background datasets for unbiased attribution.

    Parameters
    ----------
    pipeline : sklearn.Pipeline
        Trained pipeline containing a 'model' step.
    X_train : pandas.DataFrame
        Training features (unused but kept for API consistency).
    X_test : pandas.DataFrame
        Test feature matrix.
    background : numpy.ndarray
        Background dataset (cluster centers).

    Returns
    -------
    numpy.ndarray
        SHAP values for each test sample and feature.
    """
    X_test_trans = pipeline.named_steps["preprocessor"].transform(X_test)

    explainer = shap.LinearExplainer(
        pipeline.named_steps["model"],
        background
    )

    return explainer.shap_values(X_test_trans)


# ============================================================
# 3. SHAP FOR TREE-BASED MODELS
# ============================================================

def shap_tree(pipeline, X_train, X_test, background):
    """
    Compute SHAP values for tree-based models (Random Forest, XGBoost).

    Uses SHAP's TreeExplainer, which is optimized for tree ensembles
    and supports background datasets for consistent attribution.

    Parameters
    ----------
    pipeline : sklearn.Pipeline
        Trained pipeline containing a 'model' step.
    X_train : pandas.DataFrame
        Training features (unused but kept for API consistency).
    X_test : pandas.DataFrame
        Test feature matrix.
    background : numpy.ndarray
        Background dataset (cluster centers).

    Returns
    -------
    numpy.ndarray
        SHAP values for each test sample and feature.
    """
    X_test_trans = pipeline.named_steps["preprocessor"].transform(X_test)

    explainer = shap.TreeExplainer(
        pipeline.named_steps["model"],
        data=background
    )

    return explainer.shap_values(X_test_trans)


# ============================================================
# 4. SHAP SUMMARY PLOT
# ============================================================

def plot_shap_summary(shap_values, X_trans, feature_names, model_name):
    """
    Generate a standard SHAP summary plot (beeswarm).

    Parameters
    ----------
    shap_values : numpy.ndarray
        SHAP values for each sample and feature.
    X_trans : numpy.ndarray
        Transformed feature matrix.
    feature_names : list of str
        Names of transformed features.
    model_name : str
        Name of the model (used for labeling).

    Notes
    -----
    - This function displays the plot directly.
    - Saving is handled externally if needed.
    """
    shap.summary_plot(
        shap_values,
        X_trans,
        feature_names=feature_names,
        show=True
    )


# ============================================================
# 5. SELECT BEST MODEL FOR A REGIME
# ============================================================

def get_best_model_for_regime(regime_results):
    """
    Select the best-performing model for a regime based on RMSE.

    Parameters
    ----------
    regime_results : dict
        Dictionary containing:
            {
                "results": {
                    model_name: {
                        "test_metrics": {"RMSE": ...}
                    }
                }
            }

    Returns
    -------
    str
        Name of the best-performing model.
    """
    model_results = regime_results["results"]

    # Extract RMSE for each model
    rmse_scores = {
        model_name: model_results[model_name]["test_metrics"]["RMSE"]
        for model_name in model_results
    }

    # Select model with lowest RMSE
    best_model = min(rmse_scores, key=rmse_scores.get)
    return best_model

"""
SHAP Analysis — Part 2
----------------------

This section contains the main SHAP analysis workflow:

1. run_shap_for_best_model()
   - Selects the best model for a regime
   - Computes SHAP values
   - Extracts interpretability mappings
   - Generates summary, beeswarm, dependence, and grouped plots
   - Returns structured SHAP outputs

2. run_shap_all_regimes()
   - Runs SHAP analysis for all regimes

3. Helper extraction functions:
   - extract_target_encoding()
   - extract_binary_mapping()
   - extract_missing_indicator_mapping()
   - extract_numeric_summary()
   - get_common_features()
"""

def get_transformed_feature_names(preprocessor):
    """
    Returns transformed feature names after ColumnTransformer.
    """
    output_features = []

    for name, transformer, cols in preprocessor.transformers_:
        if transformer == "drop":
            continue
        if transformer == "passthrough":
            output_features.extend(cols)
        else:
            if hasattr(transformer, "get_feature_names_out"):
                fn = transformer.get_feature_names_out(cols)
                output_features.extend(fn)
            else:
                output_features.extend(cols)

    return np.array(output_features)

# ============================================================
# 6. MAIN SHAP WORKFLOW FOR A SINGLE REGIME
# ============================================================

def run_shap_for_best_model(ALL_RESULTS, regime_name, config,
                            save_dir="shap_plots", top_n=15):
    """
    Compute SHAP values for the best model in a regime and generate:
    - SHAP summary plot
    - Beeswarm plot
    - Dependence plots for top features
    - Grouped SHAP bar chart
    - Interpretability mappings (binary, target-encoded, missing indicators)
    - Numeric variable summaries

    Parameters
    ----------
    ALL_RESULTS : dict
        Full modeling results for all regimes.
    regime_name : str
        Name of the regime ("Regime A", "Regime B", etc.).
    config : dict
        GLOBAL_CONFIG dictionary.
    save_dir : str
        Directory where SHAP plots will be saved.
    top_n : int
        Number of top features to visualize in beeswarm and dependence plots.

    Returns
    -------
    dict
        {
            "best_model": ...,
            "shap_values": ...,
            "X_trans": ...,
            "feature_names": ...,
            "mean_abs_shap": ...,
            "top_features": ...,
            "top20_features": ...,
            "target_mappings": ...,
            "binary_mappings": ...,
            "missing_mappings": ...,
            "numeric_summaries": ...
        }
    """

    os.makedirs(save_dir, exist_ok=True)

    regime_data = ALL_RESULTS[regime_name]
    model_results = regime_data["results"]

    # ---------------------------------------------------------
    # 1. Identify best model
    # ---------------------------------------------------------
    best_model_name = get_best_model_for_regime(regime_data)
    print(f"\n Best model for {regime_name}: {best_model_name}")

    # Avoid SHAP on baselines
    if best_model_name in ["MedianBaseline", "DummyRegressor", "Baseline"]:
        print("Baseline model selected; overriding for SHAP.")
        for candidate in ["XGBoost", "RandomForest", "Ridge"]:
            if candidate in model_results:
                best_model_name = candidate
                print(f" Using {best_model_name} instead.")
                break

    best_model_res = model_results[best_model_name]

    pipeline = best_model_res["pipeline"]
    X_train = best_model_res["X_train"]
    X_test = best_model_res["X_test"]

    # Raw feature names AFTER dropping columns
    kept_features = best_model_res["kept_features"]

    # Preprocessor output names
    feature_names = best_model_res["feature_names"]

    # Readable names for raw kept features
    readable_kept_features = [
        feature_name_map.get(f, f) for f in kept_features
    ]

    # ---------------------------------------------------------
    # 2. Build SHAP background dataset
    # ---------------------------------------------------------
    background = build_shared_shap_background(
        pipeline,
        X_train,
        n_clusters=config["shap_background_clusters"]
    )

    # ---------------------------------------------------------
    # 3. Compute SHAP values
    # ---------------------------------------------------------
    from sklearn.dummy import DummyRegressor
    model = pipeline.named_steps["model"]

    if isinstance(model, DummyRegressor):
        raise ValueError("SHAP cannot be computed for DummyRegressor.")

    if best_model_name == "Ridge":
        shap_values = shap_linear(pipeline, X_train, X_test, background)
    else:
        shap_values = shap_tree(pipeline, X_train, X_test, background)

    print(f"\nINTERPRETABILITY EXTRACTION FOR {regime_name}")

    # ---------------------------------------------------------
    # 4. Build DataFrame with original features + target
    # ---------------------------------------------------------
    df_regime = X_test.copy()
    df_regime["working_hours"] = best_model_res["y_test"].values

    # ---------------------------------------------------------
    # 5. Extract interpretability mappings
    # ---------------------------------------------------------

    # Target-encoded variables
    target_encoded_vars = [c for c in kept_features if c.endswith("_target")]
    target_mappings = {
        var: extract_target_encoding(pipeline.named_steps["preprocessor"], var)
        for var in target_encoded_vars
    }

    # Binary variables
    binary_vars = [c for c in kept_features if c.startswith("binary__")]
    binary_mappings = {
        b: extract_binary_mapping(df_regime, shap_values, b)
        for b in binary_vars
    }

    # Missing indicators
    missing_vars = [c for c in kept_features if "missingindicator" in c]
    missing_mappings = {
        m: extract_missing_indicator_mapping(df_regime, shap_values, m)
        for m in missing_vars
    }

    # Numeric variables
    numeric_vars = [
        c for c in kept_features
        if c.startswith("num__") and "missing" not in c
    ]
    numeric_summaries = {
        n: extract_numeric_summary(df_regime, shap_values, n)
        for n in numeric_vars
    }

    # ---------------------------------------------------------
    # 6. Transform X_test and restrict to kept features
    # ---------------------------------------------------------
    X_test_trans = pipeline.named_steps["preprocessor"].transform(X_test)

    # Build raw → transformed mapping
    feature_names_arr = np.array(feature_names, dtype=str)
    
    raw_to_transformed = {}
    for raw in kept_features:
        raw_to_transformed[raw] = np.where(
            np.char.startswith(feature_names_arr, raw)
        )[0].tolist()
    
    # Drop raw features that do not map to any transformed columns
    raw_to_transformed = {
        raw: idxs
        for raw, idxs in raw_to_transformed.items()
        if len(idxs) > 0
    }
    
    if len(raw_to_transformed) == 0:
        raise ValueError("No raw features map to transformed features.")


    for raw in kept_features:
        raw_to_transformed[raw] = np.where(
            np.char.startswith(feature_names_arr, raw)
        )[0].tolist()
        
    agg_shap = aggregate_shap_by_raw_feature(
        shap_values,
        raw_to_transformed
    )


    # Build keep_idx
    keep_idx = []
    for raw in kept_features:
        keep_idx.extend(raw_to_transformed[raw])

    # Restrict SHAP matrix
    X_test_final = X_test_trans[:, keep_idx]

    # Build readable names for transformed columns
    feature_names_final = []
    for raw in kept_features:
        readable = feature_name_map.get(raw, raw)
        matches = raw_to_transformed[raw]
        feature_names_final.extend([readable] * len(matches))



    # ---------------------------------------------------------
    # 7. Beeswarm Plot (Top N Features)
    # ---------------------------------------------------------
    # Order raw features by mean absolute SHAP
    mean_abs_shap_raw = {
        raw: vals.mean() for raw, vals in agg_shap.items()
    }
    
    sorted_raw = sorted(
        mean_abs_shap_raw,
        key=mean_abs_shap_raw.get,
        reverse=True
    )

    mean_abs_shap_raw = {
        raw: np.mean(np.abs(vals))
        for raw, vals in agg_shap.items()
    }
    
    if len(mean_abs_shap_raw) == 0:
        raise ValueError(
            f"No valid SHAP values after aggregation for regime {regime_name}"
        )
    
        
    top_raw = sorted(
        mean_abs_shap_raw,
        key=mean_abs_shap_raw.get,
        reverse=True
    )[:top_n]
    
    if len(top_raw) == 0:
        raise ValueError("No features available for SHAP plotting.")
    
    X_plot = np.column_stack([
        X_test[raw].values for raw in top_raw if raw in X_test.columns
    ])
    
    shap_plot = np.column_stack([
        agg_shap[raw] for raw in top_raw
    ])
    
    feature_names_plot = [
        feature_name_map.get(raw, raw) for raw in top_raw
    ]


    plt.figure(figsize=(14, 10))
    shap.summary_plot(
        shap_values[:, top_idx],
        X_test_final[:, top_idx],
        feature_names=[feature_names_final[i] for i in top_idx],
        plot_type="dot",
        show=False
    )
    plt.title(f"{regime_name} — Beeswarm ({best_model_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{regime_name}_beeswarm.png"))
    plt.show()

    # ---------------------------------------------------------
    # 8. Return structured SHAP outputs
    # ---------------------------------------------------------
    top20_idx = sorted_idx[:20]

    top_features = [(feature_names_final[i], mean_abs[i]) for i in top_idx]
    top20_features = [(feature_names_final[i], mean_abs[i]) for i in top20_idx]

    return {
        "best_model": best_model_name,
        "shap_values": shap_values,
        "X_trans": X_test_final,
        "feature_names": feature_names_final,
        "mean_abs_shap": mean_abs,
        "top_features": top_features,
        "top20_features": top20_features,
        "target_mappings": target_mappings,
        "binary_mappings": binary_mappings,
        "missing_mappings": missing_mappings,
        "numeric_summaries": numeric_summaries
    }

def aggregate_shap_by_raw_feature(shap_values, raw_to_transformed):
    """
    Aggregate SHAP values from transformed space to raw-feature space.
    """
    agg_shap = {}
    for raw, idxs in raw_to_transformed.items():
        if len(idxs) == 0:
            continue
        agg_shap[raw] = np.abs(shap_values[:, idxs]).sum(axis=1)
    return agg_shap

# ============================================================
# 7. RUN SHAP FOR ALL REGIMES
# ============================================================

def run_shap_all_regimes(ALL_RESULTS, config, save_dir="shap_plots", top_n=15):
    """
    Run SHAP analysis for the best model in each regime.

    Parameters
    ----------
    ALL_RESULTS : dict
        Full modeling results for all regimes.
    config : dict
        GLOBAL_CONFIG dictionary.
    save_dir : str
        Directory where plots will be saved.
    top_n : int
        Number of top features to visualize.

    Returns
    -------
    dict
        {
            "Regime A": {...},
            "Regime B": {...},
            ...
        }
    """
    shap_outputs = {}

    for regime_name in ALL_RESULTS.keys():
        print("\n" + "=" * 80)
        print(f"SHAP ANALYSIS FOR {regime_name}")
        print("=" * 80)

        shap_outputs[regime_name] = run_shap_for_best_model(
            ALL_RESULTS,
            regime_name,
            config,
            save_dir=save_dir,
            top_n=top_n
        )

    return shap_outputs


# ============================================================
# 8. INTERPRETABILITY EXTRACTION HELPERS
# ============================================================

def extract_target_encoding(preprocessor, feature_name):
    """
    Extract mapping for a target-encoded categorical variable.
    """
    try:
        enc = preprocessor.named_transformers_["cat"].named_steps["targetencoder"]
        mapping = enc.mapping_
        return mapping.get(feature_name, None)
    except Exception:
        return None


def extract_binary_mapping(df, shap_values, feature, target_col="working_hours"):
    """
    Compute mean target and mean SHAP values for binary variables.
    """
    return pd.DataFrame({
        "value": [0, 1],
        "mean_target": [
            df.loc[df[feature] == 0, target_col].mean(),
            df.loc[df[feature] == 1, target_col].mean()
        ],
        "mean_shap": [
            shap_values[df[feature] == 0, df.columns.get_loc(feature)].mean(),
            shap_values[df[feature] == 1, df.columns.get_loc(feature)].mean()
        ]
    })


def extract_missing_indicator_mapping(df, shap_values, feature, target_col="working_hours"):
    """
    Same as binary mapping, but for missing indicator variables.
    """
    return extract_binary_mapping(df, shap_values, feature, target_col)


def extract_numeric_summary(df, shap_values, feature, target_col="working_hours"):
    """
    Compute descriptive statistics and SHAP correlations for numeric variables.
    """
    shap_col = shap_values[:, df.columns.get_loc(feature)]
    return {
        "describe": df[feature].describe(),
        "corr_with_target": df[[feature, target_col]].corr().iloc[0, 1],
        "corr_with_shap": np.corrcoef(df[feature], shap_col)[0, 1]
    }


def get_common_features(shap_outputs):
    """
    Identify features that appear in all regimes.
    Useful for balanced SHAP comparisons.
    """
    return sorted(
        set.intersection(
            *[set(res["feature_names"]) for res in shap_outputs.values()]
        )
    )

"""
shap_analysis.py — Part 3
-------------------------

This section contains utilities for *balanced* SHAP analysis, ensuring
fair cross‑regime comparisons by restricting SHAP computation to the
subset of features that appear in **all** regimes.

Included:

1. restrict_preprocessor()
2. run_balanced_shap_for_best_model()
3. Ranking utilities:
       • top_features_to_rank_dict()
       • build_rank_df_from_balanced_outputs()
4. Cross‑regime comparison utilities:
       • across_spearman_mean()
       • rank_variance()
       • across_regime_spearman()
       • across_regime_rank_variance()
5. rename_top_features()
"""

# ============================================================
# 1. RESTRICT PREPROCESSOR OUTPUT TO COMMON FEATURES
# ============================================================

def restrict_preprocessor(preprocessor, feature_names, keep_features):
    """
    Restrict a fitted ColumnTransformer to a subset of features by
    selecting only the transformed columns corresponding to the
    desired feature names.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted preprocessor.
    feature_names : list of str
        Names of all transformed features.
    keep_features : list of str
        Subset of features to retain.

    Returns
    -------
    list of int
        Indices of columns to keep.
    """
    keep_idx = [i for i, f in enumerate(feature_names) if f in keep_features]
    return keep_idx


# ============================================================
# 2. BALANCED SHAP ANALYSIS FOR A SINGLE REGIME
# ============================================================

def run_balanced_shap_for_best_model(
    ALL_RESULTS,
    regime_name,
    config,
    common_features,
    save_dir="shap_plots_balanced",
    top_n=15
):
    """
    Compute SHAP values using only features common across all regimes.
    Ensures fair cross‑regime interpretability.

    Steps:
    - Identify best model for the regime
    - Restrict transformed feature space to common features
    - Retrain model on restricted feature set
    - Compute SHAP values
    - Generate balanced SHAP summary & beeswarm plots
    - Print ranked SHAP features

    Parameters
    ----------
    ALL_RESULTS : dict
        Full modeling results for all regimes.
    regime_name : str
        Name of the regime.
    config : dict
        GLOBAL_CONFIG dictionary.
    common_features : list of str
        Features shared across all regimes.
    save_dir : str
        Directory for saving plots.
    top_n : int
        Number of top features to display.

    Returns
    -------
    dict
        {
            "best_model": ...,
            "shap_values": ...,
            "X_trans": ...,
            "feature_names": ...,
            "mean_abs_shap": ...,
            "top_features": ...
        }
    """
    os.makedirs(save_dir, exist_ok=True)

    regime_data = ALL_RESULTS[regime_name]
    model_results = regime_data["results"]

    # ---------------------------------------------------------
    # Select best model (override baseline if needed)
    # ---------------------------------------------------------
    best_model_name = get_best_model_for_regime(regime_data)

    if best_model_name in ["MedianBaseline", "DummyRegressor", "Baseline"]:
        for candidate in ["XGBoost", "RandomForest", "Ridge"]:
            if candidate in model_results:
                best_model_name = candidate
                break

    best_model_res = model_results[best_model_name]

    pipeline = best_model_res["pipeline"]
    X_train = best_model_res["X_train"]
    X_test = best_model_res["X_test"]
    feature_names = get_transformed_feature_names(
    pipeline.named_steps["preprocessor"]
    )

    # Apply readable names
    readable_feature_names = [
        feature_name_map.get(f, f) for f in feature_names
    ]

    # ---------------------------------------------------------
    # Transform data
    # ---------------------------------------------------------
    X_train_trans = pipeline.named_steps["preprocessor"].transform(X_train)
    X_test_trans = pipeline.named_steps["preprocessor"].transform(X_test)

    # Restrict to common features
    keep_idx = restrict_preprocessor(
        pipeline.named_steps["preprocessor"],
        feature_names,
        common_features
    )

    X_train_common = X_train_trans[:, keep_idx]
    X_test_common = X_test_trans[:, keep_idx]
    feature_names_common = [feature_names[i] for i in keep_idx]

    # Readable names for common features 
    feature_names_common_readable = [ 
        feature_name_map.get(f, f) for f in feature_names_common
    ]
    
    # ---------------------------------------------------------
    # Retrain model on restricted feature set
    # ---------------------------------------------------------
    model = pipeline.named_steps["model"].__class__(
        **pipeline.named_steps["model"].get_params()
    )
    model.fit(X_train_common, best_model_res["y_train"])

    # ---------------------------------------------------------
    # Compute SHAP values
    # ---------------------------------------------------------
    background = X_train_common[
        np.random.choice(
            X_train_common.shape[0],
            size=min(config["shap_background_clusters"], X_train_common.shape[0]),
            replace=False
        )
    ]

    if best_model_name == "Ridge":
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(X_test_common)
    else:
        explainer = shap.TreeExplainer(model, data=background)
        shap_values = explainer.shap_values(X_test_common)

    # ---------------------------------------------------------
    # Compute top features
    # ---------------------------------------------------------
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    top_idx = sorted_idx[:top_n]

    top_features = [(feature_names_common_readable[i], mean_abs[i]) for i in top_idx]

    # ---------------------------------------------------------
    # Print SHAP rankings
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"TOP {top_n} BALANCED SHAP FEATURES FOR {regime_name}")
    print("=" * 70)

    for rank, (feat, val) in enumerate(top_features, start=1):
        print(f"{rank:2d}. {feat:40s} SHAP = {val:.4f}")

    # ---------------------------------------------------------
    # Generate & save plots
    # ---------------------------------------------------------
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_test_common,
        feature_names=feature_names_common_readable,
        show=False
    )
    plt.title(f"{regime_name} — Balanced SHAP Summary")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{regime_name}_balanced_summary.png"))
    plt.show()

    # Beeswarm plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values[:, top_idx],
        X_test_common[:, top_idx],
        feature_names=[feature_names_common_readable[i] for i in top_idx],
        plot_type="dot",
        show=False
    )
    plt.title(f"{regime_name} — Balanced SHAP Beeswarm (Top {top_n})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{regime_name}_balanced_beeswarm.png"))
    plt.show()

    return {
        "best_model": best_model_name,
        "shap_values": shap_values,
        "X_trans": X_test_common,
        "feature_names": feature_names_common_readable,
        "mean_abs_shap": mean_abs,
        "top_features": top_features
    }


# ============================================================
# 3. RANKING UTILITIES
# ============================================================

def top_features_to_rank_dict(top_features):
    """
    Convert list of (feature, value) into {feature: rank}.
    """
    return {feat: rank + 1 for rank, (feat, _) in enumerate(top_features)}


def build_rank_df_from_balanced_outputs(balanced_shap_outputs):
    """
    Build a rank matrix:

    Rows   = regimes  
    Columns = features  
    Values = ranks (lower = more important)

    Missing features receive the worst rank + 1.
    """
    rank_dicts = {
        regime: top_features_to_rank_dict(info["top_features"])
        for regime, info in balanced_shap_outputs.items()
    }

    all_features = set().union(*rank_dicts.values())
    worst_rank = max(len(v) for v in rank_dicts.values()) + 1

    rank_df = pd.DataFrame([
        {f: ranks.get(f, worst_rank) for f in all_features}
        for ranks in rank_dicts.values()
    ], index=rank_dicts.keys())

    return rank_df


# ============================================================
# 4. CROSS‑REGIME COMPARISON UTILITIES
# ============================================================

def across_spearman_mean(rank_df):
    """
    Compute mean pairwise Spearman correlations between regimes.
    """
    rhos = []
    pvals = []

    for i in range(len(rank_df)):
        for j in range(i + 1, len(rank_df)):
            rho, p = spearmanr(rank_df.iloc[i], rank_df.iloc[j])
            rhos.append(rho)
            pvals.append(p)

    return {
        "mean_spearman_rho": np.mean(rhos),
        "mean_p_value": np.mean(pvals)
    }


def rank_variance(rank_df, top_n=15):
    """
    Compute variance of feature ranks across regimes.
    Lower variance = more stable feature importance.
    """
    mean_ranks = rank_df.mean()
    top_features = mean_ranks.nsmallest(top_n).index
    return rank_df[top_features].var().sort_values()


def across_regime_spearman(rank_df):
    """
    Compute pairwise Spearman correlations between all regime pairs.
    """
    rows = []
    regimes = rank_df.index.tolist()

    for i in range(len(regimes)):
        for j in range(i + 1, len(regimes)):
            r1, r2 = regimes[i], regimes[j]
            rho, p = spearmanr(rank_df.loc[r1], rank_df.loc[r2])
            rows.append({
                "Regime Pair": f"{r1} vs {r2}",
                "Spearman rho": rho,
                "p-value": p
            })

    return pd.DataFrame(rows)


def across_regime_rank_variance(rank_df, top_n=15):
    """
    Compute variance of top-N feature ranks across regimes.
    """
    mean_ranks = rank_df.mean()
    top_features = mean_ranks.nsmallest(top_n).index
    return rank_df[top_features].var().sort_values()


# ============================================================
# 5. HUMAN‑READABLE FEATURE RENAMING
# ============================================================

def rename_top_features(shap_outputs, READABLE_NAMES):
    """
    Replace raw feature names in SHAP top-features with human-readable labels.

    Parameters
    ----------
    shap_outputs : dict
        SHAP results for each regime.
    READABLE_NAMES : dict
        Mapping from raw feature names → human-readable names.

    Returns
    -------
    dict
        {
            "Regime A": [(ReadableName, shap_value), ...],
            ...
        }
    """
    renamed = {}

    for regime, res in shap_outputs.items():
        top_feats = res["top_features"]

        renamed[regime] = [
            (READABLE_NAMES.get(raw, raw), val)
            for raw, val in top_feats
        ]

    return renamed

"""
shap_analysis.py — Part 4
-------------------------

This section contains:

1. Human‑readable feature name mappings for SHAP output.
2. Utility functions for:
       • sanitizing filenames,
       • re‑plotting SHAP results with readable labels,
       • creating subgroup variables,
       • safely computing SHAP values on arbitrary samples,
       • ranking top SHAP features.

These utilities support the interpretability and reporting workflow.
"""

# ============================================================
# 1. HUMAN‑READABLE FEATURE NAME MAPPINGS
# ============================================================

READABLE_NAMES = {
    # -------------------------
    # REGIME A
    # -------------------------
    "cat_low__143_no": "No sideline job",
    "num__002": "Year of birth",
    "num__missingindicator_132": "Career satisfaction (missing)",
    "num__136": "Commute time (minutes, one-way)",
    "binary__391": "Works <36h due to family situation",
    "num__424": "Expected to work overtime",
    "binary__401": "Works <36h due to other reasons",
    "num__139": "Works evenings (6pm–midnight)",
    "005_target": "Highest completed education",
    "binary__390": "Cannot work more hours at employer",
    "num__129": "Satisfaction with working hours",
    "binary__409": "Supervises employees",
    "cat_low__402_healthcare and welfare": "Sector: healthcare & welfare",
    "num__missingindicator_412": "Work at own pace (missing)",
    "num__422": "Work gets busy",
    "num__032": "Education suitability (0–10)",
    "num__433": "Salary sufficient",
    "num__430": "Opportunity to learn new skills",
    "binary__393": "Works <36h due to activities outside home",
    "num__428": "Time pressure",

    # -------------------------
    # REGIME B
    # -------------------------
    "num__missingindicator_134": "Year entered employment (missing)",
    "binary__088": "Performs paid work",
    "num__003": "Age",
    "num__missingindicator_136": "Commute time (missing)",
    "num__517": "Minimum wage offer (euros)",
    "num__missingindicator_517": "Minimum wage offer (missing)",
    "num__420": "Mental effort required",
    "num__133": "Satisfaction with current work",
    "num__missingindicator_138": "Work outside office hours (missing)",
    "num__missingindicator_426": "Overall job satisfaction (missing)",

    # -------------------------
    # REGIME C
    # -------------------------
    "611_target": "Occupation (ISCO-08)",
    "num__610": "Hours worked at home",
    "num__309": "Preferred weekly hours (if part-time)",
    "binary__572": "Lives with partner",
    "cat_low__604_missing": "Partner income replacement (missing)",
    "num__130": "Satisfaction with type of work",
    "num__435": "Job insecurity",

    # Duplicates across regimes
    "binary__088": "Performs paid work",
    "binary__391": "Works <36h due to family situation",
    "binary__401": "Works <36h due to other reasons",
}


# ============================================================
# 2. FILENAME SANITIZATION
# ============================================================

def sanitize_filename(name):
    """
    Replace illegal filename characters with underscores.

    Useful when saving SHAP plots that include feature names.
    """
    return re.sub(r'[<>:"/\\|?*]', '_', name)


# ============================================================
# 3. REPLOT SHAP WITH HUMAN‑READABLE FEATURE NAMES
# ============================================================

def replot_shap_with_readable_names(raw_shap_outputs, READABLE_NAMES):
    """
    Re‑plot SHAP visualizations (beeswarm + bar plots) for each regime
    using human‑readable feature names.

    Parameters
    ----------
    raw_shap_outputs : dict
        {
            "Regime A": {
                "shap_values": ...,
                "X_trans": ...,
                "feature_names": [...]
            },
            ...
        }

    READABLE_NAMES : dict
        Mapping from raw feature names → readable labels.

    Notes
    -----
    - Uses plt.close() to avoid memory warnings.
    """

    for regime, shap_info in raw_shap_outputs.items():

        readable = [READABLE_NAMES.get(f, f) for f in shap_info["feature_names"]]

        # -----------------------------
        # Beeswarm plot
        # -----------------------------
        plt.figure(figsize=(14, 10))
        shap.summary_plot(
            shap_info["shap_values"],
            shap_info["X_trans"],
            feature_names=readable,
            show=False
        )
        plt.title(f"SHAP Beeswarm — {regime}")
        plt.tight_layout()
        plt.savefig(f"shap_{regime}_beeswarm.png")
        plt.close()

        # -----------------------------
        # Bar plot
        # -----------------------------
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_info["shap_values"],
            shap_info["X_trans"],
            feature_names=readable,
            plot_type="bar",
            show=False
        )
        plt.title(f"SHAP Bar — {regime}")
        plt.tight_layout()
        plt.savefig(f"shap_{regime}_bar.png")
        plt.close()


# ============================================================
# 4. CREATE SUBGROUP VARIABLES FOR SUBGROUP SHAP
# ============================================================

def create_subgroups(df):
    """
    Create derived subgroup variables on top of the original survey dataframe.
    Used for subgroup‑specific SHAP analysis.

    Includes:
    - Age groups
    - Sector groups
    - Life situation
    - Education groups
    """
    df = df.copy()

    # -----------------------------
    # Age groups
    # -----------------------------
    if '003' in df.columns:
        df['age_group'] = pd.cut(
            df['003'],
            bins=[0, 30, 50, 100],
            labels=['Young', 'Mid-career', 'Older']
        )

    # -----------------------------
    # Sector grouping
    # -----------------------------
    if '402' in df.columns:
        sector_groups = {
    1: "Agriculture & Extraction",   # agriculture, forestry, fishery, hunting
    2: "Agriculture & Extraction",   # mining

    3: "Production & Construction",  # industrial production
    4: "Production & Construction",  # utilities production/distribution/trade
    5: "Production & Construction",  # construction

    6: "Trade & Market Services",    # retail trade
    7: "Trade & Market Services",    # catering
    8: "Trade & Market Services",    # transport, storage, communication
    9: "Trade & Market Services",    # financial
    10: "Trade & Market Services",   # business services

    11: "Public & Social Services",  # government services
    12: "Public & Social Services",  # education
    13: "Public & Social Services",  # healthcare and welfare

    14: "Culture & Other Services",  # environmental services, culture, recreation
    15: "Culture & Other Services"   # other

        }
        df['sector_group'] = df['402'].map(sector_groups)

    # -----------------------------
    # Life situation
    # -----------------------------
    if '436' in df.columns and '450' in df.columns:
        df['life_situation'] = np.where(
            (df['436'] == 1) | (df['450'] == 1),
            'Provides care',
            'None'
        )

    # -----------------------------
    # Education grouping
    # -----------------------------
    education_groupings = {
        'STEM': ['018', '019', '025'],
        'Business_Law': ['016', '017'],
        'Arts_Humanities': ['013', '014', '015'],
        'Health_Care': ['021', '022'],
        'Services': ['023', '024', '026'],
        'Education': ['011', '012'],
        'Agriculture': ['020']
    }

    for group, cols in education_groupings.items():
        existing = [c for c in cols if c in df.columns]
        df[group] = df[existing].sum(axis=1) if existing else 0

    df['education_group'] = df[list(education_groupings.keys())].idxmax(axis=1)

    return df


# ============================================================
# 5. SAFE SHAP COMPUTATION FOR ARBITRARY SAMPLES
# ============================================================

def compute_shap_values_safe(model_pipeline, X_sample):
    """
    Align a sample dataframe with the fitted pipeline's preprocessor
    and compute SHAP values on the transformed data.

    Useful for subgroup SHAP or scenario analysis.
    """
    X_sample = X_sample.copy()

    # Ensure categorical columns are strings
    cat_cols = X_sample.select_dtypes(include=['object', 'category']).columns
    X_sample[cat_cols] = X_sample[cat_cols].astype(str)

    model = model_pipeline.named_steps['model']
    preprocessor = model_pipeline.named_steps['preprocessor']

    # Align columns
    expected_cols = preprocessor.feature_names_in_
    missing_cols = set(expected_cols) - set(X_sample.columns)
    for col in missing_cols:
        X_sample[col] = 0

    X_sample_aligned = X_sample[expected_cols]
    X_transformed = preprocessor.transform(X_sample_aligned)

    # Feature names
    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = expected_cols

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    return shap_values, X_transformed, feature_names


# ============================================================
# 6. RANK TOP FEATURES
# ============================================================

def rank_top_features(shap_values, feature_names, top_n=15):
    """
    Rank features by mean absolute SHAP value.

    Returns a DataFrame with:
    - Feature
    - MeanAbsSHAP
    - Rank
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    min_len = min(len(feature_names), len(mean_abs_shap))

    df = pd.DataFrame({
        'Feature': feature_names[:min_len],
        'MeanAbsSHAP': mean_abs_shap[:min_len]
    })

    df = df.sort_values('MeanAbsSHAP', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1

    return df.head(top_n)
"""
shap_analysis.py — Part 5
-------------------------

This section contains utilities for **subgroup-level SHAP analysis**.
These functions allow you to:

1. Compute SHAP values separately for each subgroup
   (e.g., age groups, sectors, life situations, education groups).

2. Compare SHAP rankings:
       • within a regime (across subgroups)
       • across regimes (for the same subgroup)
       • across regimes (for all subgroups)

3. Compute Spearman correlations between subgroup SHAP rankings.

This enables fairness analysis, heterogeneity analysis, and
cross‑regime interpretability.
"""

# ============================================================
# 1. SHAP ANALYSIS FOR A SINGLE SUBGROUP
# ============================================================

def shap_subgroup_analysis(
    model_pipeline,
    df,
    subgroup_col,
    sample_size=1000,
    top_n=15,
    save_dir="shap_subgroups",
    feature_names_global=None
):
    """
    Run SHAP analysis within each subgroup of a given column
    for a single regime.

    Only subgroups with sufficient sample size are analyzed.

    Parameters
    ----------
    model_pipeline : sklearn.Pipeline
        Trained pipeline containing preprocessor + model.
    df : pandas.DataFrame
        Full regime dataset (after subgroup creation).
    subgroup_col : str
        Column defining subgroups (e.g., "age_group").
    sample_size : int
        Maximum number of samples per subgroup.
    top_n : int
        Number of top features to extract.
    save_dir : str
        Directory for saving subgroup SHAP plots.
    feature_names_global : list or None
        Optional global feature name list to enforce consistency.

    Returns
    -------
    dict
        {
            subgroup_name: DataFrame of top features,
            ...
        }
    """
    os.makedirs(save_dir, exist_ok=True)

    if subgroup_col not in df.columns:
        print(f"Column '{subgroup_col}' not found. Skipping.")
        return {}

    subgroups = df[subgroup_col].dropna().unique()
    if len(subgroups) == 0:
        print(f"No valid subgroups in '{subgroup_col}'. Skipping.")
        return {}

    results = {}

    for subgroup in subgroups:
        print(f"\nSHAP Analysis for subgroup: {subgroup}")
        df_sub = df[df[subgroup_col] == subgroup].copy()

        # Require minimum sample size
        if len(df_sub) < 300:
            print(f"Skipping {subgroup} (too few samples: {len(df_sub)})")
            continue

        # Sample within subgroup
        X_sample = df_sub.sample(
            min(sample_size, len(df_sub)),
            random_state=42
        )

        # Compute SHAP values safely
        shap_vals, X_transformed, feature_names_raw = compute_shap_values_safe(
            model_pipeline, X_sample
        )

        # Use global feature names if provided
        feature_names = feature_names_global or feature_names_raw

        # Rank features
        top_features = rank_top_features(shap_vals, feature_names, top_n)
        results[subgroup] = top_features

        # Beeswarm plot
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[-top_n:]

        shap.summary_plot(
            shap_vals[:, top_idx],
            X_transformed[:, top_idx],
            feature_names=[readable_feature_names[i] for i in top_idx],
            plot_type="dot",
            show=False
        )

        plt.xlabel("SHAP Value (Impact on Prediction)")
        plt.title(f"{subgroup}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{subgroup}_beeswarm.png"))
        plt.show()

    return results


# ============================================================
# 2. RUN SUBGROUP SHAP FOR A SINGLE REGIME
# ============================================================

def run_shap_subgroups_for_regime(
    regime_name,
    df_final,
    model_result,
    sample_size=1000,
    top_n=15
):
    """
    Perform subgroup SHAP analysis for a single regime.

    Parameters
    ----------
    regime_name : str
        Name of the regime.
    df_final : pandas.DataFrame
        Full regime dataset (before subgroup creation).
    model_result : dict
        Model result dictionary from ALL_RESULTS[regime]["results"][model].
    sample_size : int
        Max samples per subgroup.
    top_n : int
        Number of top features to extract.

    Returns
    -------
    dict
        {
            "age_group": {...},
            "sector_group": {...},
            ...
        }
    """
    print(f"\n==============================")
    print(f" SHAP SUBGROUP ANALYSIS — {regime_name}")
    print(f"==============================")

    # Create subgroup labels AFTER global SHAP
    df_sub = create_subgroups(df_final.copy())

    model_pipeline = model_result["pipeline"]
    feature_names = model_result["feature_names"]

    results = {}

    # Age groups
    results["age_group"] = shap_subgroup_analysis(
        model_pipeline, df_sub, "age_group",
        sample_size=sample_size, top_n=top_n,
        save_dir=f"shap_{regime_name}_age",
        feature_names_global=feature_names
    )

    # Sector groups
    results["sector_group"] = shap_subgroup_analysis(
        model_pipeline, df_sub, "sector_group",
        sample_size=sample_size, top_n=top_n,
        save_dir=f"shap_{regime_name}_sector",
        feature_names_global=feature_names
    )

    # Life situation
    results["life_situation"] = shap_subgroup_analysis(
        model_pipeline, df_sub, "life_situation",
        sample_size=sample_size, top_n=top_n,
        save_dir=f"shap_{regime_name}_life",
        feature_names_global=feature_names
    )

    # Education groups
    results["education_group"] = shap_subgroup_analysis(
        model_pipeline, df_sub, "education_group",
        sample_size=sample_size, top_n=top_n,
        save_dir=f"shap_{regime_name}_education",
        feature_names_global=feature_names
    )

    return results


# ============================================================
# 3. RANK DICTIONARY UTILITIES
# ============================================================

def subgroup_shap_to_rank_dict(subgroup_shap_df):
    """
    Convert a subgroup SHAP top-features DataFrame into:
    {feature_name: rank}
    """
    return {
        row["Feature"]: row["Rank"]
        for _, row in subgroup_shap_df.iterrows()
    }


def spearman_between_rank_dicts(rank_dict_1, rank_dict_2):
    """
    Compute Spearman correlation between two subgroup rank dictionaries.

    Missing features receive the worst rank.
    """
    all_features = set(rank_dict_1.keys()) | set(rank_dict_2.keys())
    worst_rank = max(len(rank_dict_1), len(rank_dict_2)) + 1

    vec1 = [rank_dict_1.get(f, worst_rank) for f in all_features]
    vec2 = [rank_dict_2.get(f, worst_rank) for f in all_features]

    rho, p = spearmanr(vec1, vec2)
    return rho, p


# ============================================================
# 4. SPEARMAN WITHIN A SINGLE REGIME
# ============================================================

def spearman_within_regime(subgroup_results):
    """
    Compute Spearman correlations between all subgroup pairs
    within a regime.

    Returns
    -------
    DataFrame with:
    Subgroup Pair | Spearman rho | p-value
    """
    rows = []
    subgroups = list(subgroup_results.keys())

    for i in range(len(subgroups)):
        for j in range(i + 1, len(subgroups)):

            s1, s2 = subgroups[i], subgroups[j]
            df1 = subgroup_results[s1]
            df2 = subgroup_results[s2]

            rank1 = subgroup_shap_to_rank_dict(df1)
            rank2 = subgroup_shap_to_rank_dict(df2)

            rho, p = spearman_between_rank_dicts(rank1, rank2)
            rows.append([f"{s1} vs {s2}", rho, p])

    return pd.DataFrame(rows, columns=["Subgroup Pair", "Spearman rho", "p-value"])


# ============================================================
# 5. DIRECT COMPARISON OF SUBGROUPS WITHIN A REGIME
# ============================================================

def compare_subgroups_within_regime(shap_results_regime):
    """
    Compare SHAP rankings across subgroups within a single regime.

    Parameters
    ----------
    shap_results_regime : dict
        {
            "age_group": DataFrame,
            "sector_group": DataFrame,
            ...
        }

    Returns
    -------
    DataFrame
        Subgroup Pair | Spearman rho | p-value
    """
    subgroup_names = list(shap_results_regime.keys())
    rows = []

    for i in range(len(subgroup_names)):
        for j in range(i + 1, len(subgroup_names)):

            s1, s2 = subgroup_names[i], subgroup_names[j]
            df1 = shap_results_regime[s1]
            df2 = shap_results_regime[s2]

            merged = df1.merge(df2, on="Feature", suffixes=("_1", "_2"))
            rho, p = spearmanr(merged["Rank_1"], merged["Rank_2"])

            rows.append({
                "Subgroup Pair": f"{s1} vs {s2}",
                "Spearman rho": rho,
                "p-value": p
            })

    return pd.DataFrame(rows)


# ============================================================
# 6. SPEARMAN ACROSS REGIMES (ALL SUBGROUPS)
# ============================================================

def spearman_across_regimes(subgroup_results_a, subgroup_results_b, subgroup_results_c):
    """
    Compute Spearman correlations for each subgroup across regimes.

    Returns
    -------
    DataFrame with:
    Subgroup | Regime Pair | Spearman rho | p-value
    """
    rows = []

    all_subgroups = (
        set(subgroup_results_a.keys()) |
        set(subgroup_results_b.keys()) |
        set(subgroup_results_c.keys())
    )

    for subgroup in all_subgroups:

        df_a = subgroup_results_a.get(subgroup)
        df_b = subgroup_results_b.get(subgroup)
        df_c = subgroup_results_c.get(subgroup)

        rank_a = subgroup_shap_to_rank_dict(df_a) if df_a is not None else None
        rank_b = subgroup_shap_to_rank_dict(df_b) if df_b is not None else None
        rank_c = subgroup_shap_to_rank_dict(df_c) if df_c is not None else None

        # A vs B
        if rank_a and rank_b:
            rho, p = spearman_between_rank_dicts(rank_a, rank_b)
            rows.append([subgroup, "A vs B", rho, p])

        # A vs C
        if rank_a and rank_c:
            rho, p = spearman_between_rank_dicts(rank_a, rank_c)
            rows.append([subgroup, "A vs C", rho, p])

        # B vs C
        if rank_b and rank_c:
            rho, p = spearman_between_rank_dicts(rank_b, rank_c)
            rows.append([subgroup, "B vs C", rho, p])

    return pd.DataFrame(rows, columns=["Subgroup", "Regime Pair", "Spearman rho", "p-value"])


# ============================================================
# 7. COMPARE SAME SUBGROUP ACROSS REGIMES
# ============================================================

def compare_subgroup_across_regimes(shap_results_a, shap_results_b, shap_results_c, subgroup_name):
    """
    Compare the SAME subgroup across Regime A, B, C.

    Example:
        compare_subgroup_across_regimes(results_A, results_B, results_C, "Young")

    Returns
    -------
    DataFrame with:
    Subgroup | Regime Pair | Spearman rho | p-value
    """
    dfs = []
    labels = []

    for regime_name, results in zip(
        ["Regime A", "Regime B", "Regime C"],
        [shap_results_a, shap_results_b, shap_results_c]
    ):
        for group_type, group_dict in results.items():
            if subgroup_name in group_dict:
                dfs.append(group_dict[subgroup_name])
                labels.append(regime_name)

    rows = []

    for i in range(len(dfs)):
        for j in range(i + 1, len(dfs)):

            df1, df2 = dfs[i], dfs[j]
            r1, r2 = labels[i], labels[j]

            merged = df1.merge(df2, on="Feature", suffixes=("_1", "_2"))
            rho, p = spearmanr(merged["Rank_1"], merged["Rank_2"])

            rows.append({
                "Subgroup": subgroup_name,
                "Regime Pair": f"{r1} vs {r2}",
                "Spearman rho": rho,
                "p-value": p
            })

    return pd.DataFrame(rows)
"""
shap_analysis.py — Part 6
-------------------------

This final section provides utilities for *global subgroup comparison*:

1. compute_all_subgroup_spearman()
   - Computes Spearman correlations:
       • across regimes (A–B, A–C, B–C)
       • within each regime (subgroup vs subgroup)

2. table_all_subgroups_across_regimes()
   - Builds a unified table listing ALL subgroups across ALL subgroup
     categories (age, sector, life situation, education) and their
     cross‑regime Spearman correlations.

These functions support high‑level interpretability comparisons and
heterogeneity analysis across demographic and occupational groups.
"""

# ============================================================
# 1. COMPUTE SPEARMAN CORRELATIONS FOR ALL SUBGROUP CATEGORIES
# ============================================================

def compute_all_subgroup_spearman(shap_results_a, shap_results_b, shap_results_c):
    """
    Compute Spearman correlations for each subgroup category.

    For each category (e.g., "age_group", "sector_group"), compute:
    - Across-regime correlations (A vs B, A vs C, B vs C)
    - Within-regime correlations (subgroup vs subgroup inside A, B, C)

    Parameters
    ----------
    shap_results_a : dict
        Subgroup SHAP results for Regime A.
    shap_results_b : dict
        Subgroup SHAP results for Regime B.
    shap_results_c : dict
        Subgroup SHAP results for Regime C.

    Returns
    -------
    dict
        {
            "age_group": {
                "across_regimes": DataFrame,
                "within_regime_A": DataFrame,
                "within_regime_B": DataFrame,
                "within_regime_C": DataFrame
            },
            ...
        }
    """
    results = {}

    for subgroup_category in shap_results_a.keys():

        across = spearman_across_regimes(
            shap_results_a[subgroup_category],
            shap_results_b[subgroup_category],
            shap_results_c[subgroup_category]
        )

        within_A = spearman_within_regime(shap_results_a[subgroup_category])
        within_B = spearman_within_regime(shap_results_b[subgroup_category])
        within_C = spearman_within_regime(shap_results_c[subgroup_category])

        results[subgroup_category] = {
            "across_regimes": across,
            "within_regime_A": within_A,
            "within_regime_B": within_B,
            "within_regime_C": within_C
        }

    return results


# ============================================================
# 2. BUILD A GLOBAL TABLE OF ALL SUBGROUPS ACROSS REGIMES
# ============================================================

def table_all_subgroups_across_regimes(shap_results_a, shap_results_b, shap_results_c):
    """
    Build a single table listing ALL subgroups (across all subgroup categories)
    and their Spearman correlations across Regime A, B, C.

    Parameters
    ----------
    shap_results_a : dict
        Subgroup SHAP results for Regime A.
    shap_results_b : dict
        Subgroup SHAP results for Regime B.
    shap_results_c : dict
        Subgroup SHAP results for Regime C.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - Subgroup Category
        - Subgroup
        - Regime Pair (A vs B, A vs C, B vs C)
        - Spearman rho
        - p-value
    """
    rows = []

    # Loop over subgroup categories: age_group, sector_group, etc.
    for subgroup_category in shap_results_a.keys():

        sub_a = shap_results_a[subgroup_category]
        sub_b = shap_results_b[subgroup_category]
        sub_c = shap_results_c[subgroup_category]

        # All subgroup names inside this category
        all_subgroups = set(sub_a.keys()) | set(sub_b.keys()) | set(sub_c.keys())

        for subgroup in all_subgroups:

            df_a = sub_a.get(subgroup)
            df_b = sub_b.get(subgroup)
            df_c = sub_c.get(subgroup)

            rank_a = subgroup_shap_to_rank_dict(df_a) if df_a is not None else None
            rank_b = subgroup_shap_to_rank_dict(df_b) if df_b is not None else None
            rank_c = subgroup_shap_to_rank_dict(df_c) if df_c is not None else None

            # A vs B
            if rank_a and rank_b:
                rho, p = spearman_between_rank_dicts(rank_a, rank_b)
                rows.append([subgroup_category, subgroup, "A vs B", rho, p])

            # A vs C
            if rank_a and rank_c:
                rho, p = spearman_between_rank_dicts(rank_a, rank_c)
                rows.append([subgroup_category, subgroup, "A vs C", rho, p])

            # B vs C
            if rank_b and rank_c:
                rho, p = spearman_between_rank_dicts(rank_b, rank_c)
                rows.append([subgroup_category, subgroup, "B vs C", rho, p])

    return pd.DataFrame(
        rows,
        columns=["Subgroup Category", "Subgroup", "Regime Pair", "Spearman rho", "p-value"]
    )
