"""
column_cleaning.py
------------------

This module contains a collection of low-level cleaning utilities used
throughout the LISS data preprocessing pipeline. These functions remove
irrelevant variables, drop empty columns, normalize text, convert
string-based missing values, and eliminate high-cardinality or
administrative variables that are unsuitable for modeling.

Each function is designed to be:
- modular,
- non-destructive (returns a copy),
- reusable across multiple datasets and years.

This file is imported by the main cleaning pipeline and by notebooks.
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd      # DataFrame operations
import numpy as np       # Numerical operations, NaN handling
import re                # Regular expressions (used in some cleaning logic)
import ast               # Safely evaluate string representations of lists/dicts


# ============================================================
# COLUMN REMOVAL UTILITIES
# ============================================================

def drop_columns(df, cols_to_remove):
    """
    Remove a predefined list of unwanted columns.

    This function is typically used early in the cleaning pipeline to
    eliminate variables that are known to be irrelevant, redundant,
    or structurally unusable (e.g., encrypted identifiers, metadata,
    or administrative codes).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    cols_to_remove : list of str
        Column names that should be removed if present.

    Returns
    -------
    df_out : pandas.DataFrame
        A new DataFrame with the specified columns removed.
    removed_cols : list of str
        The subset of `cols_to_remove` that actually existed in the DataFrame.

    Notes
    -----
    - The function is non-destructive (returns a copy).
    - Only columns that exist in the DataFrame are removed.
    - Useful for enforcing a consistent schema across datasets.
    """
    df = df.copy()

    # remove any column that CONTAINS any of the patterns
    to_drop = [col for col in df.columns if any(p in col for p in cols_to_remove)]

    df = df.drop(columns=to_drop)

    return df, to_drop


def drop_empty_columns(df):
    """
    Remove columns that contain only missing values.

    This step helps reduce dimensionality and prevents issues during
    preprocessing, modeling, and SHAP analysis. Columns that are fully
    empty across all rows provide no information and should be removed.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    df_out : pandas.DataFrame
        A new DataFrame with empty columns removed.
    empty_cols : list of str
        Names of columns that were entirely NA.

    Notes
    -----
    - Non-destructive: returns a new DataFrame.
    - Helps avoid dtype inference warnings during concatenation.
    - Reduces memory usage and fragmentation.
    """
    df = df.copy()

    # Identify columns where *all* values are NaN
    empty_cols = df.columns[df.isna().all()].tolist()

    # Drop them
    df = df.drop(columns=empty_cols)

    return df, empty_cols



def drop_pension_exit_and_text_columns(df):
    """
    Remove pension-related variables, early-exit variables,
    and Dutch free-text/high-cardinality columns.

    These variables are typically unsuitable for regression modeling
    because they:
    - contain long free-text responses,
    - have extremely high cardinality,
    - represent administrative pension codes irrelevant to your target,
    - or introduce noise and instability into SHAP explanations.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    df_out : pandas.DataFrame
        A new DataFrame with pension, exit, and text columns removed.
    removed_cols : list of str
        Columns that were actually removed.

    Notes
    -----
    - Column names are normalized to string to avoid dtype mismatches.
    - Pension variables span multiple numeric ranges (146–287).
    - Dutch text columns are manually curated based on LISS documentation.
    - This step significantly reduces dimensionality and improves model stability.
    """
    df = df.copy()

    # Ensure all column names are strings (LISS sometimes mixes int/str)
    df.columns = df.columns.astype(str)

    # Pensioner indicator
    pension_cols = ['099', '99']

    # Pension received (146–187)
    pension_received_cols = [f"{i:03d}" for i in range(146, 188)]

    # Pension funds (188–287)
    pension_funds_cols = [f"{i:03d}" for i in range(188, 288)]

    # Early exit (290–307)
    early_exit_cols = [f"{i:03d}" for i in range(290, 308)]

    # Dutch free-text / high-cardinality columns
    dutch_text_cols = [
        '224', '225', '317', '165', '160', '164', '156', '152', '148',
        '112', '082', '085', '071', '065', '058', '054', '048', '041',
        '037', '029', '010', '007'
    ]

    # Additional known irrelevant columns
    extra_remove_cols = ['522', '523', '126']

    # Combine all removal lists
    cols_to_remove = (
        pension_cols +
        pension_received_cols +
        pension_funds_cols +
        early_exit_cols +
        dutch_text_cols +
        extra_remove_cols
    )

    # Keep only columns that exist in the dataset
    existing = [c for c in cols_to_remove if c in df.columns]

    # Drop them
    df_clean = df.drop(columns=existing)

    return df_clean, existing



def drop_contract_hours_and_income(df):
    """
    Remove contract hours and income variables.

    Contract hours ('126') and income variables (576–578) are removed
    because they:
    - may leak target information,
    - are inconsistently measured across years,
    - or introduce multicollinearity and instability in SHAP values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    df_out : pandas.DataFrame
        A new DataFrame with contract hours and income variables removed.
    removed_cols : list of str
        Columns that were actually removed.

    Notes
    -----
    - Column names are normalized to string for consistency.
    - Income variables are zero-padded to match LISS naming conventions.
    - This step is typically applied after removing pension variables.
    """
    df = df.copy()
    df.columns = df.columns.astype(str)

    # Contract hours (potential leakage)
    contract_hours_cols = ['126']

    # Income variables (576–578)
    income_cols = [f"{i:03d}" for i in range(576, 579)]

    # Employment Type & features related to preferred working hours
    employment_hours_cols = ['121', '380', '145']

    cols_to_remove = contract_hours_cols + income_cols + employment_hours_cols

    # Keep only columns that exist
    existing = [c for c in cols_to_remove if c in df.columns]

    df_clean = df.drop(columns=existing)

    return df_clean, existing


# ============================================================
# TEXT NORMALIZATION & STRING-BASED MISSING VALUES
# ============================================================

def replace_string_nan(df):
    """
    Convert string-based missing values into actual np.nan.

    Many LISS variables contain string placeholders such as:
    - "nan"
    - "none"
    - "geen" (Dutch: "none")
    - "geen antwoord" (Dutch: "no answer")

    This function standardizes all of them into np.nan.
    """
    df = df.copy()

    # Only apply to object/string columns
    object_cols = df.select_dtypes(include="object").columns

    # Set of string values that should be treated as missing
    string_nan_values = {
        "nan", "none", "null", "n/a", "na", "nvt",
        "geen", "geen antwoord", "no answer"
    }

    for col in object_cols:
        df[col] = df[col].apply(
            lambda x: np.nan
            if isinstance(x, str) and x.strip().lower() in string_nan_values
            else x
        )

    return df


def normalize_all_text(df):
    """
    Normalize all object-type columns by stripping whitespace
    and converting text to lowercase.

    This ensures consistent formatting across years and prevents
    issues during category matching, merging, and encoding.
    """
    df = df.copy()

    # Identify all text columns
    object_cols = df.select_dtypes(include="object").columns

    def normalize_value(x):
        # Only modify strings; leave numbers untouched
        return x.strip().lower() if isinstance(x, str) else x

    # Apply normalization to each column
    for col in object_cols:
        df[col] = df[col].apply(normalize_value)

    return df

# ============================================================
# STRING NORMALIZATION: "DON'T KNOW" / REFUSAL RESPONSES
# ============================================================

def replace_dont_know(df):
    """
    Replace 'don't know' and 'prefer not to say' responses with np.nan.

    Many LISS survey variables contain free-text responses that indicate
    uncertainty or refusal to answer. These responses are semantically
    equivalent to missing values and should be standardized to np.nan
    to avoid polluting categorical encodings or numeric conversions.

    This function:
    - normalizes text to lowercase,
    - checks against a curated set of "don't know" and "prefer not to say" variants,
    - replaces all matches with np.nan.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    df_out : pandas.DataFrame
        Dataset with uncertainty/refusal responses replaced by np.nan.
    """
    df = df.copy()
    object_cols = df.select_dtypes(include="object").columns

    # Variants of "I don't know" (English + Dutch + encoding issues)
    DONT_KNOW = {
        "i don't know", "i dont know", "i don t know", "i dontknow",
        "i don’t know", "i don´t know", "idk", "dont know", "don't know",
        "i don't know anymore", "i dont know anymore",
        "i don’t know (anymore)", "unknown", "not known",
        "geen idee", "weet ik niet", "weet niet",
        "i don't have plans", "i dont have plans",
        "i don't know/i don't have any plans",
        "i don’t know/i don’t have any plans",
        "i don’t know / i don’t have plans",
        "i don't know / i don't have plans",
        "not applicable", "not relevant", "i don\x92t know"
    }

    # Variants of refusal responses
    PREFER_NOT = {
        "i prefer not to say", "prefer not to say",
        "i dont want to say", "i don't want to say",
        "rather not say"
    }

    # Combine and normalize to lowercase
    REPLACE_SET = {v.lower() for v in (DONT_KNOW | PREFER_NOT)}

    def normalize_and_check(x):
        if isinstance(x, str):
            cleaned = x.strip().lower()
            return np.nan if cleaned in REPLACE_SET else x
        return x

    # Apply to all object columns
    for col in object_cols:
        df[col] = df[col].apply(normalize_and_check)

    return df


# ============================================================
# VERBAL SCALE CONVERSION (0–10, 1–5)
# ============================================================

def apply_verbal_scales(df):
    """
    Convert verbal 0–10 and 1–5 scales into numeric values.

    LISS surveys often encode satisfaction, suitability, or problem severity
    using verbal labels that correspond to numeric scales. This function
    standardizes those labels into numeric values so they can be used in
    regression models and SHAP analysis.

    Examples:
    - "does not at all suit my work" → 0
    - "suits my work perfectly" → 10
    - "certainly not" → 1
    - "certainly yes" → 5

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    df_out : pandas.DataFrame
        Dataset with verbal scales converted to numeric dtype.
    """
    df = df.copy()
    object_cols = df.select_dtypes(include="object").columns

    # Verbal equivalents for 0–10 scale
    scale_0_10 = {
        "does not at all suit my work": "0",
        "0 = does not at all suit my work": "0",
        "suits my work perfectly": "10",
        "10 = suits my work perfectly": "10",
        "not at all satisfied": "0",
        "0 = not at all satisfied": "0",
        "fully satisfied": "10",
        "10 = fully satisfied": "10",
        "very serious problems": "0",
        "0 = very serious problems": "0",
        "no problems at all": "10",
        "10 = no problems at all": "10"
    }

    # Verbal equivalents for 1–5 Likert scale
    likert_1_5 = {
        "certainly not": "1",
        "1 = certainly not": "1",
        "certainly yes": "5",
        "5 = certainly yes": "5"
    }

    # Merge both dictionaries into a single lookup table
    REPLACE_MAP = {
        **{k.lower(): v for k, v in scale_0_10.items()},
        **{k.lower(): v for k, v in likert_1_5.items()}
    }

    def replace_value(x):
        if isinstance(x, str):
            cleaned = x.strip().lower()
            return REPLACE_MAP.get(cleaned, x)
        return x

    # Apply replacement and attempt numeric conversion
    for col in object_cols:
        df[col] = df[col].apply(replace_value)
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            # Some columns contain mixed types; ignore conversion errors
            pass

    return df


# ============================================================
# BINARY YES/NO → 1/0
# ============================================================

def convert_binary(df):
    """
    Convert yes/no columns into binary 1/0.

    This function automatically detects columns where all non-null
    values are either "yes" or "no" (case-insensitive), and converts
    them into integer-coded binary variables.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    df_out : pandas.DataFrame
        Dataset with binary columns encoded as Int64.
    """
    df = df.copy()
    object_cols = df.select_dtypes(include="object").columns

    for col in object_cols:
        # Normalize text
        cleaned = df[col].apply(
            lambda x: x.strip().lower() if isinstance(x, str) else x
        )

        non_null = cleaned.dropna()

        # Check if column is strictly yes/no
        if non_null.isin(["yes", "no"]).all():
            df[col] = cleaned.map({"yes": 1, "no": 0}).astype("Int64")

    return df


# ============================================================
# NUMERIC-LIKE STRINGS → NUMERIC DTYPE
# ============================================================

def convert_numeric_like(df):
    """
    Convert numeric-like strings (e.g., '12', '3.5', '4,2') into numeric dtype.

    This function detects columns where all non-empty values match a numeric
    pattern and converts them into float or integer dtype. It also handles
    European decimal commas by replacing ',' with '.'.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    df_out : pandas.DataFrame
        Dataset with numeric-like columns converted to numeric dtype.
    """
    df = df.copy()
    object_cols = df.select_dtypes(include="object").columns

    # Regex for integers or decimals with . or ,
    numeric_pattern = re.compile(r"^[+-]?\d+([.,]\d+)?$")

    def is_numeric_like(x):
        return isinstance(x, str) and bool(numeric_pattern.match(x.strip()))

    for col in object_cols:
        cleaned = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

        # Remove empty strings and NaN
        non_empty = cleaned.dropna()
        non_empty = non_empty[non_empty != ""]

        # If all values match numeric pattern → convert
        if len(non_empty) > 0 and non_empty.apply(is_numeric_like).all():
            cleaned = cleaned.apply(
                lambda x: x.replace(",", ".") if isinstance(x, str) else x
            )
            df[col] = pd.to_numeric(cleaned, errors="coerce")

    return df


# ============================================================
# ORDINAL SCALES (FREQUENCY, AGREEMENT, MONTHS)
# ============================================================

def convert_ordinal_scales(df):
    """
    Convert ordinal verbal scales (frequency, agreement, months) into numeric codes.

    This function detects columns that contain:
    - frequency scales: ["never", "sometimes", "often"]
    - agreement scales: ["disagree entirely", "disagree", "agree", "agree entirely"]
    - month names: january → 1, ..., december → 12
    - binary lists: ["0", "1"]

    It automatically maps them to ordered numeric values.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    df_out : pandas.DataFrame
        Dataset with ordinal scales converted to numeric dtype.
    """
    df = df.copy()
    object_cols = df.select_dtypes(include="object").columns

    # Known ordinal scales
    frequency_scale = ["never", "sometimes", "often"]
    agreement_scale = ["disagree entirely", "disagree", "agree", "agree entirely"]

    # Month mapping
    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12
    }

    def parse_value(x):
        """
        Convert stringified lists (e.g., "[yes, no]") into Python lists.
        """
        if isinstance(x, list):
            return x
        if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
            try:
                return ast.literal_eval(x)
            except Exception:
                return x
        return x

    for col in object_cols:
        s = df[col]

        # Parse list-like values
        parsed = s.apply(parse_value)

        # Flatten all values into a single list for scale detection
        flattened = []
        for val in parsed.dropna():
            flattened.extend(val if isinstance(val, list) else [val])

        # Normalize values
        flattened = [str(v).strip().lower() for v in flattened if v not in ["", None]]
        uniques = sorted(set(flattened))

        # Frequency scale
        if set(uniques) == set(frequency_scale):
            order = {"never": 1, "sometimes": 2, "often": 3}
            df[col] = parsed.apply(
                lambda x: order.get(x[0].lower()) if isinstance(x, list)
                else order.get(str(x).strip().lower())
            ).astype("Int64")
            continue

        # Agreement scale
        if set(uniques) == set(agreement_scale):
            order = {
                "disagree entirely": 1,
                "disagree": 2,
                "agree": 3,
                "agree entirely": 4
            }
            df[col] = parsed.apply(
                lambda x: order.get(x[0].lower()) if isinstance(x, list)
                else order.get(str(x).strip().lower())
            ).astype("Int64")
            continue

        # Month names
        if set(uniques).issubset(month_map.keys()):
            df[col] = parsed.apply(
                lambda x: month_map.get(str(x).strip().lower())
            ).astype("Int64")
            continue

        # Binary lists (e.g., ["0", "1"])
        if set(uniques).issubset({"0", "1", 0, 1}):
            df[col] = parsed.apply(
                lambda x: int(x[0]) if isinstance(x, list) else int(x)
            ).astype("Int64")
            continue

    return df
