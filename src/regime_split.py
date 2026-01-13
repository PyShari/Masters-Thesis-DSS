"""
regime_split.py
----------------

This module provides utilities for:

1. Filtering respondents based on key inclusion criteria.
2. Splitting the dataset into three economic regimes:
       • Regime A: 2008–2013  (Post-crisis, pre-gig economy)
       • Regime B: 2014–2019  (Stable growth, pre-COVID)
       • Regime C: 2020–2024  (COVID + remote work era)
3. Summarizing regime characteristics.

These functions are used early in the pipeline to structure the dataset
into meaningful economic periods for modeling and analysis.
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import re   # Included for consistency, though not used directly here


# ============================================================
# 1. FILTER RESPONDENTS BY INCLUSION CRITERIA
# ============================================================

def filter_by_conditions(df):
    """
    Filter respondents based on key inclusion criteria.

    Inclusion rule:
    - Respondent is included if:
        • Column '088' == 1   (e.g., employed)
          OR
        • Column '102' == 1   (e.g., self-employed)

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataset.

    Returns
    -------
    pandas.DataFrame
        Filtered dataset containing only respondents who meet
        at least one of the inclusion criteria.

    Notes
    -----
    - This function is non-destructive (returns a copy).
    - Column names are assumed to follow LISS numeric-string format.
    """

    df = df.copy()

    if "001" not in df.columns:
        raise ValueError("Column '001' not found in DataFrame.")

    mask = df["001"] == 1
    return df[mask].copy()

# ============================================================
# 2. SPLIT DATASET INTO ECONOMIC REGIMES
# ============================================================

def split_by_regime(df):
    """
    Split the filtered dataset into three economic regimes.

    Regime definitions:
    - Regime A: 2008–2013  (Post-crisis, pre-gig economy)
    - Regime B: 2014–2019  (Stable growth, pre-COVID)
    - Regime C: 2020–2024  (COVID + remote work era)

    Parameters
    ----------
    df : pandas.DataFrame
        Filtered dataset containing a 'year' column.

    Returns
    -------
    dict
        {
            "A": DataFrame for Regime A,
            "B": DataFrame for Regime B,
            "C": DataFrame for Regime C
        }

    Notes
    -----
    - The function assumes that the dataset contains a 'year' column.
    - Each regime is returned as a separate DataFrame.
    """
    df = df.copy()

    regime_a = df[(df["year"] >= 2008) & (df["year"] <= 2013)].copy()
    regime_b = df[(df["year"] >= 2014) & (df["year"] <= 2019)].copy()
    regime_c = df[(df["year"] >= 2020) & (df["year"] <= 2024)].copy()

    return {
        "A": regime_a,
        "B": regime_b,
        "C": regime_c
    }


# ============================================================
# 3. SUMMARIZE REGIME CHARACTERISTICS
# ============================================================

def summarize_regimes(regimes):
    """
    Print summary information for each regime.

    Parameters
    ----------
    regimes : dict
        Output from split_by_regime(), mapping regime labels to DataFrames.

    Prints
    ------
    - Unique years present in each regime
    - Shape of each regime DataFrame

    Notes
    -----
    - Useful for sanity checks before modeling.
    """
    for key, df in regimes.items():
        years = sorted(df["year"].unique())
        print(f"Regime {key}: years={years}, shape={df.shape}")
