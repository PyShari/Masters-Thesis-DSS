"""
cleaning_pipeline.py
--------------------

This module defines the `clean_full_dataset` function, which orchestrates the
entire multi-layer cleaning pipeline used throughout the project. It combines
general text normalization, missing-value handling, scale conversions, 
column-specific overrides, and dataset-level standardization into one unified 
function.

This file is designed to be imported into notebooks and scripts to ensure that 
all datasets undergo the exact same cleaning steps in a reproducible manner.
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd     # Core library for DataFrame manipulation
import numpy as np      # Numerical operations and NaN handling
import re               # Regular expressions (used in some cleaning utilities)


# ------------------------------------------------------------
# Import general-purpose column cleaning utilities
# These functions perform broad transformations across many columns.
# ------------------------------------------------------------
from src.column_cleaning import (
    normalize_all_text,          # Standardizes text: lowercase, strip whitespace, unify formatting
    replace_string_nan,          # Converts string-based "nan", "NA", etc. into actual np.nan
    replace_dont_know,           # Replaces "don't know" / "refuse" responses with np.nan
    apply_verbal_scales,         # Converts verbal Likert-type scales into numeric equivalents
    convert_binary,              # Converts yes/no or similar binary fields into 0/1
    convert_numeric_like,        # Converts numeric-like strings ("12", "003", "5.0") into numeric dtype
    convert_ordinal_scales       # Converts ordinal categories into ordered numeric scales
)


# ------------------------------------------------------------
# Import override rules
# These functions apply targeted corrections to specific columns
# or perform dataset-wide normalization.
# ------------------------------------------------------------
from src.column_overrides import (
    clean_selected_columns,      # Applies manual cleaning rules to known problematic columns
    clean_remaining_columns,     # Applies fallback cleaning rules to all other columns
    clean_dataset,               # Performs dataset-level normalization (e.g., renaming, harmonizing)
    normalize_regime             # Ensures regime labels are standardized and consistent
)


# ============================================================
# MAIN CLEANING FUNCTION
# ============================================================

def clean_full_dataset(df):
    """
    Clean an entire LISS dataset using a structured, multi-layer pipeline.

    Parameters
    ----------
    df : pandas.DataFrame
        The raw or partially processed dataset that needs to be cleaned.

    Returns
    -------
    pandas.DataFrame
        A fully cleaned and standardized dataset ready for analysis,
        modeling, or regime splitting.

    Cleaning Pipeline Overview
    --------------------------
    The cleaning process is intentionally layered to ensure clarity,
    reproducibility, and modularity:

    **Layer 1 — General text & value cleaning**
        - Normalize text formatting
        - Convert string-based missing values to NaN
        - Replace "don't know" responses
        - Convert verbal scales to numeric
        - Convert binary fields to 0/1
        - Convert numeric-like strings to numeric dtype
        - Convert ordinal scales to ordered numeric values

    **Layer 2 — Column-specific overrides**
        - Apply custom cleaning rules to known problematic columns
        - Fix inconsistent coding, mislabeled categories, or special cases

    **Layer 3 — Remaining problematic columns**
        - Apply fallback cleaning rules to all columns not covered in Layer 2

    **Layer 4 — Dataset-level normalization**
        - Apply final dataset-wide cleaning (e.g., harmonizing column names)
        - Normalize regime labels for consistency across years

    This layered approach ensures that:
        - Broad cleaning happens first
        - Specific fixes override general rules
        - Dataset-wide consistency is enforced last
    """

    # Work on a copy to avoid modifying the original DataFrame
    df = df.copy()

    # --------------------------------------------------------
    # Layer 1 — General text cleaning & type conversions
    # --------------------------------------------------------
    df = normalize_all_text(df)          # Standardize all text fields
    df = replace_string_nan(df)          # Convert "nan"/"NA" strings to np.nan
    df = replace_dont_know(df)           # Replace "don't know" responses
    df = apply_verbal_scales(df)         # Convert verbal Likert scales to numeric
    df = convert_binary(df)              # Convert binary yes/no fields to 0/1
    df = convert_numeric_like(df)        # Convert numeric-like strings to numbers
    df = convert_ordinal_scales(df)      # Convert ordinal categories to numeric order

    # --------------------------------------------------------
    # Layer 2 — Column-specific overrides
    # --------------------------------------------------------
    df = clean_selected_columns(df)      # Apply manual corrections to known columns

    # --------------------------------------------------------
    # Layer 3 — Remaining problematic columns
    # --------------------------------------------------------
    df = clean_remaining_columns(df)     # Apply fallback cleaning to all other columns

    # --------------------------------------------------------
    # Layer 4 — Dataset-level normalization
    # --------------------------------------------------------
    df = clean_dataset(df)               # Apply final dataset-wide cleaning rules
    df = normalize_regime(df)            # Standardize regime labels (A/B/C)

    return df
