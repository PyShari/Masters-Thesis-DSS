"""
eda_utils.py
------------

This module provides a collection of exploratory data analysis (EDA)
utilities used to inspect, diagnose, and visualize the LISS datasets
and the regime‑specific subsets.

It includes:
- basic dataset inspection,
- structured regime inspection,
- skewness and outlier diagnostics,
- correlation and VIF analysis,
- sorted correlation pair extraction,
- target variable summary and distribution plotting.

These functions are designed to be imported into notebooks for
interactive exploration and reporting.
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd          # Tabular data manipulation
import numpy as np           # Numerical operations
import seaborn as sns        # Statistical plotting
import matplotlib.pyplot as plt  # Visualization

from scipy.stats import skew, kurtosis   # Distribution shape metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
# VIF is used to detect multicollinearity among numeric predictors


# ============================================================
# BASIC DATAFRAME INFORMATION
# ============================================================

def basic_info(df, name):
    """
    Print a quick overview of a dataset:
    - shape
    - dtypes
    - column names
    - first rows

    Useful for sanity checks before deeper EDA.
    """
    print(f"\n\n===== {name} =====")
    print("Shape:", df.shape)

    print("\n--- Dtypes ---")
    print(df.dtypes)

    print("\n--- Columns ---")
    print(list(df.columns))

    print("\n--- First Rows ---")
    print(df.head())


# ============================================================
# 1. STRUCTURED INSPECTION OF A REGIME DATASET
# ============================================================

def inspect_regime(df, name="Regime"):
    """
    Perform a structured exploratory inspection of a dataset.

    Includes:
    - shape
    - column types
    - missingness percentages
    - numeric descriptive statistics
    - categorical descriptive statistics
    - unique value counts
    - sample rows
    - constant columns (no variation)
    - columns with >90% missing values

    Returns a dictionary with:
    - missingness series
    - constant columns list
    - high-missing columns list
    """

    print(f"\n===== {name} Shape =====")
    print(df.shape)

    print(f"\n===== {name} Column Types =====")
    print(df.dtypes)

    print(f"\n===== {name} Missingness Percent =====")
    missing = df.isna().mean().sort_values(ascending=False) * 100
    print(missing)

    print(f"\n===== {name} Descriptive Statistics (Numeric) =====")
    print(df.describe())

    print(f"\n===== {name} Descriptive Statistics (Categorical) =====")
    print(df.describe(include='object'))

    print(f"\n===== {name} Number of Unique Values per Column =====")
    print(df.nunique().sort_values())

    print(f"\n===== {name} Sample Rows =====")
    print(df.head())

    # Columns with only one unique value (constant)
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    print(f"\n===== {name} Columns with Only One Unique Value =====")
    print(constant_cols)

    # Columns with >90% missing values
    high_missing = missing[missing > 90]
    print(f"\n===== {name} Columns with More Than Ninety Percent Missing =====")
    print(high_missing)

    return {
        "missing": missing,
        "constant_cols": constant_cols,
        "high_missing": high_missing
    }


# ============================================================
# 2. FULL EDA: SKEWNESS, OUTLIERS, CORRELATION, VIF
# ============================================================

def run_full_eda(df, name="Regime"):
    """
    Perform exploratory diagnostics on numeric columns:

    - Skewness: detects asymmetry in distributions
    - Outlier counts: IQR-based outlier detection
    - Correlation matrix: linear relationships
    - VIF values: multicollinearity diagnostics

    Returns a dictionary with:
    - skewness series
    - outlier counts
    - correlation matrix
    - VIF DataFrame
    """

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # -----------------------------
    # Skewness
    # -----------------------------
    print(f"\n===== {name} Skewness =====")
    skew_vals = df[numeric_cols].skew()
    print(skew_vals)

    # -----------------------------
    # Outlier detection (IQR rule)
    # -----------------------------
    print(f"\n===== {name} Outlier Summary =====")
    outlier_counts = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_counts[col] = ((df[col] < lower) | (df[col] > upper)).sum()
    print(pd.Series(outlier_counts))

    # -----------------------------
    # Correlation matrix
    # -----------------------------
    print(f"\n===== {name} Correlation Matrix =====")
    corr_matrix = df[numeric_cols].corr()
    print(corr_matrix)

    # -----------------------------
    # VIF (Variance Inflation Factor)
    # -----------------------------
    print(f"\n===== {name} VIF Values =====")
    vif_data = []
    clean_df = df[numeric_cols].dropna()

    for i, col in enumerate(numeric_cols):
        try:
            vif = variance_inflation_factor(clean_df.values, i)
        except Exception:
            vif = np.nan
        vif_data.append((col, vif))

    vif_df = pd.DataFrame(vif_data, columns=["Variable", "VIF"])
    print(vif_df)

    return {
        "skewness": skew_vals,
        "outliers": pd.Series(outlier_counts),
        "correlation": corr_matrix,
        "vif": vif_df
    }


# ============================================================
# 3. SORTED CORRELATION PAIRS
# ============================================================

def list_sorted_correlations(corr_matrix, min_abs_corr=0.0):
    """
    Convert a correlation matrix into a sorted list of feature pairs.

    Steps:
    - Extract upper triangle of correlation matrix
    - Convert to long format
    - Filter by minimum absolute correlation
    - Sort by absolute correlation descending

    Returns a DataFrame with:
    - Feature_1
    - Feature_2
    - Correlation
    """

    corr_pairs = (
        corr_matrix
        .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )

    corr_pairs.columns = ["Feature_1", "Feature_2", "Correlation"]

    # Filter by threshold
    corr_pairs = corr_pairs[corr_pairs["Correlation"].abs() >= min_abs_corr]

    # Sort by absolute correlation
    corr_pairs = corr_pairs.reindex(
        corr_pairs["Correlation"].abs().sort_values(ascending=False).index
    )

    return corr_pairs


# ============================================================
# 4. TARGET SUMMARY + DISTRIBUTION PLOT
# ============================================================

REGIME_COLORS = {
    "Regime A": "#1f77b4",
    "Regime B": "#ff7f0e",
    "Regime C": "#2ca02c"
}

sns.set_theme(style="whitegrid", palette="muted")


def target_summary(df, target, regime_label):
    """
    Print summary statistics and plot the distribution of a target variable.

    Includes:
    - descriptive statistics
    - missingness summary
    - skewness & kurtosis
    - extreme value count (IQR rule)
    - histogram + KDE plot
    - automatic saving of PNG and PDF versions
    """

    # Determine color based on regime
    regime_key = regime_label.split("(")[0].strip()
    color = REGIME_COLORS.get(regime_key, "#333333")

    print(f"\nSummary statistics for '{target}' in {regime_label}:")
    print(df[target].describe())

    # -----------------------------
    # Missingness summary
    # -----------------------------
    total = len(df)
    valid = df[target].notna().sum()
    missing = total - valid

    print(f"\nTotal respondents: {total}")
    print(f"Valid {target}: {valid}")
    print(f"Missing {target}: {missing}")
    print(f"Valid %: {valid/total*100:.2f}")

    # -----------------------------
    # Distribution shape
    # -----------------------------
    valid_values = df[target].dropna()
    print(f"\nSkewness: {skew(valid_values):.2f}")
    print(f"Kurtosis: {kurtosis(valid_values):.2f}")

    # -----------------------------
    # Extreme values (IQR rule)
    # -----------------------------
    q1 = valid_values.quantile(0.25)
    q3 = valid_values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    extremes = valid_values[(valid_values < lower) | (valid_values > upper)]
    print(f"\nExtreme values: {len(extremes)}")

    # -----------------------------
    # Plot distribution
    # -----------------------------
    plt.figure(figsize=(10, 6))
    sns.histplot(valid_values, bins=30, kde=True, color=color, alpha=0.6)

    plt.title(f"Distribution of {target} – {regime_label}", fontsize=14, weight='bold')
    plt.xlabel(target)
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Safe filename
    safe_label = (
        regime_label.replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("–", "-")
    )

    plt.savefig(f"{target}_{safe_label}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{target}_{safe_label}.pdf", dpi=300, bbox_inches="tight")

    plt.tight_layout()
    plt.show()
