"""
column_overrides.py — Selected Column Cleaning Rules
----------------------------------------------------

This module defines `clean_selected_columns`, a targeted cleaning function
that applies **column-specific overrides** for variables that require
special handling beyond the general cleaning pipeline.

These overrides are necessary because certain LISS variables:
- use inconsistent coding across years,
- contain verbal scales that must be mapped to numeric values,
- include Dutch text responses,
- encode yes/no in multiple formats,
- store months as text,
- contain numeric-like strings that must be converted,
- or include special “don’t know” categories.

This function is imported by the main cleaning pipeline and ensures that
known problematic columns are cleaned consistently across all datasets.
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd      # DataFrame operations
import numpy as np       # Numerical operations, NaN handling
import re                # Regular expressions for numeric detection


# ============================================================
# MAIN FUNCTION: CLEAN SELECTED COLUMNS
# ============================================================

def clean_selected_columns(df):
    """
    Apply column-specific cleaning rules for variables that require
    special handling beyond the general text-cleaning pipeline.

    This function performs targeted cleaning for known problematic
    LISS variables. It includes:

    - Converting categorical dtype → string
    - Text normalization (lowercase, strip whitespace)
    - Replacing string-based NaNs and "don't know" responses
    - Mapping yes/no columns to binary 1/0
    - Converting 0–10 Likert scales to numeric
    - Converting frequency scales (never/sometimes/often)
    - Converting agreement scales (disagree → agree entirely)
    - Converting month names to numeric (January = 1)
    - Converting numeric-like strings to numeric dtype
    - Handling special “don’t know” columns

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    df_out : pandas.DataFrame
        Dataset with selected columns cleaned using override rules.
    """
    df = df.copy()

    # ---------------------------------------------------------
    # 0. Convert categorical columns to string
    # ---------------------------------------------------------
    # Some LISS variables are stored as pandas "category" dtype.
    # Converting them to string ensures consistent downstream processing.
    cat_cols = df.select_dtypes(include="category").columns
    df[cat_cols] = df[cat_cols].astype("string")

    # ---------------------------------------------------------
    # 1. TEXT NORMALIZATION
    # ---------------------------------------------------------
    # Standardize all string values: lowercase + strip whitespace.
    df = df.apply(
        lambda col: col.map(
            lambda x: x.strip().lower() if isinstance(x, str) else x
        )
    )

    # ---------------------------------------------------------
    # 2. STRING NANS + DON'T KNOW
    # ---------------------------------------------------------
    # Replace known variants of "don't know" or refusal responses with NaN.
    dont_know = {
        "i don't know","i dont know","i don’t know","i dontknow",
        "i don t know","unknown","not known","geen idee",
        "weet ik niet","weet niet","i don’t know (anymore)",
        "i don't know anymore","i prefer not to say",
        "prefer not to say","i dont want to say",
        "i don't want to say","rather not say",
        "= 'i don't know'","= 'i prefer not to say'",
        "i don’t know"
    }

    df = df.apply(lambda col: col.where(~col.isin(dont_know), np.nan))

    # ---------------------------------------------------------
    # 3. YES/NO BINARY COLUMNS
    # ---------------------------------------------------------
    # These columns consistently encode yes/no responses.
    yes_no_cols = [
        "000","028","111","200","211","216","223","287","307",
        "455","456","457","458","459","460","464","465","466",
        "467","470","479","478","483","485","486","487","488",
        "491","492","493","494","495","556"
    ]

    def yes_no_map(x):
        if isinstance(x, str):
            if x in ["yes", "yes, does work"]:
                return 1
            if x in ["no", "no, does not work"]:
                return 0
        return x

    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map(yes_no_map).astype("Int64")

    # ---------------------------------------------------------
    # 4. 0–10 LIKERT SCALES
    # ---------------------------------------------------------
    # These columns encode importance/suitability on a 0–10 scale.
    likert_cols = ["032", "070", "087"]

    likert_map = {
        "not at all important": 0,
        "0 = not at all important": 0,
        "do not suit my work at all": 0,
        "0 = do not suit my work at all": 0,
        "extremely important": 10,
        "10 = extremely important": 10,
        "suit my work perfectly": 10,
        "10 = suit my work perfectly": 10
    }

    numeric_pattern = re.compile(r"^[+-]?\d+([.,]\d+)?$")

    def convert_likert(x):
        if isinstance(x, str):
            # Map verbal labels
            if x in likert_map:
                return likert_map[x]
            # Convert numeric-like strings
            if numeric_pattern.match(x):
                return float(x.replace(",", "."))
        return x

    for col in likert_cols:
        if col in df.columns:
            df[col] = df[col].map(convert_likert).astype("Float64")

    # ---------------------------------------------------------
    # 5. FREQUENCY SCALES
    # ---------------------------------------------------------
    freq_cols = ["419","420","421","422","423","424","425"]
    freq_map = {"never": 1, "sometimes": 2, "often": 3}

    for col in freq_cols:
        if col in df.columns:
            df[col] = df[col].map(freq_map).astype("Int64")

    # ---------------------------------------------------------
    # 6. AGREEMENT SCALES
    # ---------------------------------------------------------
    agree_cols = ["426","427","428","429","430","431","432","433","434","435"]
    agree_map = {
        "disagree entirely": 1,
        "disagree": 2,
        "agree": 3,
        "agree entirely": 4
    }

    for col in agree_cols:
        if col in df.columns:
            df[col] = df[col].map(agree_map).astype("Int64")

    # ---------------------------------------------------------
    # 7. MONTH COLUMNS
    # ---------------------------------------------------------
    # Convert month names to numeric month index.
    month_cols = ["135","320","341","441","443","453","471"]

    month_map = {
        "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
        "july":7,"august":8,"september":9,"october":10,"november":11,"december":12
    }

    for col in month_cols:
        if col in df.columns:
            df[col] = df[col].map(month_map).astype("Int64")

    # ---------------------------------------------------------
    # 8. NUMERIC-LIKE STRINGS
    # ---------------------------------------------------------
    numeric_like_cols = ["032","070","087","314"]

    def numeric_like(x):
        if isinstance(x, str):
            # Convert numeric-like strings
            if numeric_pattern.match(x):
                return float(x.replace(",", "."))
            # Mixed alphanumeric → treat as missing
            if any(c.isdigit() for c in x) and any(c.isalpha() for c in x):
                return np.nan
        return x

    for col in numeric_like_cols:
        if col in df.columns:
            df[col] = df[col].map(numeric_like).astype("Float64")

    # ---------------------------------------------------------
    # 9. SPECIAL DON'T-KNOW COLUMNS
    # ---------------------------------------------------------
    # These columns contain many "don't know" variants.
    dont_know_cols = ["312","365","468","531","532","533","534","613"]

    for col in dont_know_cols:
        if col in df.columns:
            df[col] = df[col].where(~df[col].isin(dont_know), np.nan)

    return df

# ============================================================
# CLEAN REMAINING COLUMNS — FALLBACK CLEANING RULES
# ============================================================

def clean_remaining_columns(df):
    """
    Apply fallback cleaning rules to all remaining columns that were not
    handled by `clean_selected_columns`.

    This function is designed to catch:
    - leftover categorical columns,
    - general text normalization,
    - additional "don't know" variants,
    - ordinal mappings for specific variables,
    - Likert scale corrections,
    - time-unit normalization,
    - frequency scales,
    - work schedule scales,
    - binary yes/no variants,
    - special multi-category collapses.

    It ensures that *all* columns follow consistent formatting and
    numeric encoding before the dataset enters the modeling pipeline.
    """
    df = df.copy()

    # ---------------------------------------------------------
    # 0. Convert categorical columns to string
    # ---------------------------------------------------------
    # Ensures consistent downstream processing.
    cat_cols = df.select_dtypes(include="category").columns
    df[cat_cols] = df[cat_cols].astype("string")

    # ---------------------------------------------------------
    # 1. Normalize text (lowercase + strip whitespace)
    # ---------------------------------------------------------
    df = df.apply(
        lambda col: col.map(
            lambda x: x.strip().lower() if isinstance(x, str) else x
        )
    )

    # ---------------------------------------------------------
    # 2. Replace "don't know" variants with NaN
    # ---------------------------------------------------------
    # Captures additional uncertainty/refusal responses.
    dont_know = {
        "i don't know","i dont know","i don’t know","i dontknow",
        "i don t know","unknown","not known","geen idee",
        "weet ik niet","weet niet","i don’t know (anymore)",
        "i don't know anymore","i prefer not to say",
        "prefer not to say","i dont want to say",
        "i don't want to say","rather not say"
    }

    df = df.apply(lambda col: col.where(~col.isin(dont_know), np.nan))

    # ---------------------------------------------------------
    # 3. Ordinal mapping for variables 031 and 033
    # ---------------------------------------------------------
    # These variables encode how well education matches current work.
    ordinal_031_033 = {
        "has no relation at all to my current work": 0,
        "have no relation at all to my current work": 0,
        "has become outdated because the work has changed": 1,
        "have become outdated because the work has changed": 1,
        "is insufficiently geared to the work practice": 2,
        "are insufficiently geared to the work practice": 2,
        "is lower than the level required by my work": 3,
        "are lower than the level required by my work": 3,
        "is approximately at the level required by my work": 4,
        "are approximately at the level required by my work": 4,
        "is higher than the level required by my work": 5,
        "are higher than the level required by my work": 5,
        "is for another kind of work than for my current work": 6,
        "are for another kind of work than for my current work": 6
    }

    for col in ["031", "033"]:
        if col in df.columns:
            df[col] = df[col].map(ordinal_031_033).astype("Int64")

    # ---------------------------------------------------------
    # 4. Fix Likert scale in variable 053 (0–10)
    # ---------------------------------------------------------
    likert_map_053 = {
        "not at all important": 0,
        "0 = not at all important": 0,
        "0 not at all important": 0,
        "extremely important": 10,
        "10 = extremely important": 10,
        "10 extremely important": 10
    }

    numeric_pattern = re.compile(r"^[+-]?\d+([.,]\d+)?$")

    def convert_053(x):
        if isinstance(x, str):
            if x in likert_map_053:
                return likert_map_053[x]
            if numeric_pattern.match(x):
                return float(x.replace(",", "."))
        return np.nan

    if "053" in df.columns:
        df["053"] = df["053"].map(convert_053).astype("Float64")

    # ---------------------------------------------------------
    # 5. Normalize time units in variable 077 → convert to days
    # ---------------------------------------------------------
    time_map = {
        "days": 1,
        "part-days": 0.5,
        "weeks": 7,
        "months": 30,
        "years": 365
    }

    if "077" in df.columns:
        df["077"] = df["077"].map(time_map).astype("Float64")

    # ---------------------------------------------------------
    # 6. Frequency scales (412–418)
    # ---------------------------------------------------------
    freq_map = {"never": 1, "sometimes": 2, "often": 3}

    for col in ["412","413","414","415","416","417","418"]:
        if col in df.columns:
            df[col] = df[col].map(freq_map).astype("Int64")

    # ---------------------------------------------------------
    # 7. Work schedule ordinal scales (138–143)
    # ---------------------------------------------------------
    work_map = {
        "no": 0,
        "yes, i work in shifts": 1,
        "yes, i often work outside regular office hours": 2,
        "yes, i (almost) always work in the evening or at night": 3
    }

    if "138" in df.columns:
        df["138"] = df["138"].map(work_map).astype("Int64")

    # 139–141: frequency-like work schedule
    freq4_map = {
        "no, i never work in the evening": 1,
        "i rarely work in the evening": 2,
        "i work one or more evenings once every few weeks": 3,
        "i work one or more evenings almost every week": 4
    }

    if "139" in df.columns:
        df["139"] = df["139"].map(freq4_map).astype("Int64")

    # ---------------------------------------------------------
    # 8. Yes/no + variants → binary
    # ---------------------------------------------------------
    yes_no_cols = ["310","365","468","531","532","533","613"]

    def yes_no_binary(x):
        if isinstance(x, str):
            if x == "yes":
                return 1
            if x == "no":
                return 0
        return np.nan

    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map(yes_no_binary).astype("Int64")

    # ---------------------------------------------------------
    # 9. Collapse "probably yes/no" categories (variable 538)
    # ---------------------------------------------------------
    if "538" in df.columns:
        df["538"] = df["538"].map({
            "yes": 1,
            "probably yes": 1,
            "no": 0,
            "probably not": 0
        }).astype("Int64")

    # ---------------------------------------------------------
    # 10. Multi-category mapping for variable 612
    # ---------------------------------------------------------
    if "612" in df.columns:
        df["612"] = df["612"].map({
            "no": 0,
            "yes but this has not been granted": 1,
            "yes and this has been granted": 2
        }).astype("Int64")

    return df


# ============================================================
# DATASET-LEVEL NORMALIZATION
# ============================================================

def clean_dataset(df):
    """
    Apply dataset-wide cleaning rules that operate after all
    column-level transformations are complete.

    This includes:
    - fixing Likert scales for variables 128–133,
    - converting nullable Int64 columns to float64,
    - converting yes/no variables to binary,
    - cleaning "don't know" responses,
    - ensuring all object columns become pandas string dtype.

    These steps ensure consistent datatypes across all years.
    """
    df = df.copy()

    # ============================================
    # 1. Fix Likert scales for variables 128–133
    # ============================================
    likert_10_cols = ["128", "129", "130", "131", "133"]

    def clean_likert_10(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        if x.startswith("10"):
            return 10.0
        if x.startswith("0"):
            return 0.0
        if x.isdigit():
            return float(x)
        return np.nan

    for col in likert_10_cols:
        df[col] = df[col].apply(clean_likert_10).astype("float64")

    # ============================================
    # 2. Convert all Int64 → float64
    # ============================================
    int64_cols = df.select_dtypes(include=["Int64"]).columns
    df[int64_cols] = df[int64_cols].astype("float64")

    # ============================================
    # 3. Convert Yes/No variables → 1/0
    # ============================================
    yes_no_cols = [
        "260", "282", "283", "284", "285", "361",
        "364", "388", "389", "512", "513", "514", "515", "325"
    ]
    
    def yes_no_to_binary(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip().lower()
        if x == "yes":
            return 1.0
        if x == "no":
            return 0.0
        return np.nan
    
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].apply(yes_no_to_binary).astype("float64")

    # ============================================
    # 4. Fix "I don't know" / "I prefer not to say"
    # ============================================
    dk_cols = ["323", "324"]

    def clean_dontknow(x):
        if pd.isna(x):
            return np.nan
        x = str(x).lower().strip()
        if "i don't know" in x or "i prefer not to say" in x:
            return np.nan
        return x

    for col in dk_cols:
        df[col] = df[col].apply(clean_dontknow)

    # ============================================
    # 5. Convert all remaining objects → strings
    # ============================================
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].astype("string")

    return df


# ============================================================
# FINAL NORMALIZATION OF REGIME DATASETS
# ============================================================

def normalize_regime(df):
    """
    Final normalization step applied after splitting the dataset
    into economic regimes.

    Ensures:
    - nullable Float64 → float64,
    - pandas string dtype → object (for parquet compatibility),
    - pd.NA → np.nan everywhere.

    This guarantees consistent datatypes across all regime datasets.
    """
    df = df.copy()

    # 1. Convert nullable Float64 → float64
    float64_nullable = df.select_dtypes(include=["Float64"]).columns
    df[float64_nullable] = df[float64_nullable].astype("float64")

    # 2. Convert pandas string dtype → object
    string_cols = df.select_dtypes(include=["string"]).columns
    df[string_cols] = df[string_cols].astype("object")

    # 3. Replace pd.NA → np.nan everywhere
    df = df.replace({pd.NA: np.nan})

    return df
