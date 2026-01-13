"""
dataset_cleaning.py
-------------------

This module provides utilities for cleaning and standardizing the raw
LISS survey datasets before merging them into a unified panel.

It includes:

1. Prefix removal (LISS often prefixes variables with wave identifiers)
2. String normalization (lowercase, trimmed)
3. Year extraction and cleaning
4. Column deduplication
5. Removal of empty columns
6. Dataset‑level processing
7. Merging multiple yearly datasets into one panel

These steps ensure consistent variable naming, prevent merge conflicts,
and prepare the data for downstream preprocessing and modeling.
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd


# ============================================================
# 1. REMOVE PREFIXES
# ============================================================

def remove_prefixes(df):
    """
    Remove the first 5 characters from each column name.

    LISS datasets often include prefixes like "p123_" or "w2010_".
    This function strips those prefixes to standardize column names.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Copy of df with cleaned column names.
    """
    df = df.copy()
    df.columns = [col[5:] if len(col) > 5 else col for col in df.columns]
    return df


# ============================================================
# 2. CLEAN YEAR COLUMN
# ============================================================

def clean_year_column(df, colname="year"):
    """
    Extract a 4‑digit year from a string column and convert to Int64.

    Handles cases like:
        "year_2012"
        "wave2014"
        "2013 survey"

    Parameters
    ----------
    df : pandas.DataFrame
    colname : str
        Column containing year information.

    Returns
    -------
    pandas.DataFrame
    """
    df = df.copy()
    df[colname] = df[colname].astype(str)

    df[colname] = (
        df[colname]
        .str.findall(r"\d{4}")          # extract all 4‑digit sequences
        .apply(lambda x: x[-1] if len(x) > 0 else None)  # take last match
        .astype("Int64")
    )
    return df


# ============================================================
# 3. DEDUPLICATE COLUMN NAMES
# ============================================================

def deduplicate_columns(columns):
    """
    Ensure column names are unique by appending suffixes.

    Example:
        ["age", "age", "income"] → ["age", "age_1", "income"]

    Parameters
    ----------
    columns : list of str

    Returns
    -------
    list of str
        Deduplicated column names.
    """
    seen = {}
    new_cols = []

    for col in columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")

    return new_cols


# ============================================================
# 4. STANDARDIZE STRING COLUMNS
# ============================================================

def standardize_strings(df):
    """
    Strip whitespace and convert all string columns to lowercase.

    Ensures consistent categorical encoding across years.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip().str.lower()

    return df


# ============================================================
# 5. PROCESS A SINGLE YEARLY DATASET
# ============================================================

def process_dataset(df, year):
    """
    Apply all cleaning steps to a single dataset.

    Steps:
    - Remove prefixes
    - Standardize string columns
    - Add year column (e.g., "year_2012")
    - Drop columns that are entirely empty
    - Deduplicate column names
    - Reset index

    Parameters
    ----------
    df : pandas.DataFrame
    year : int or str
        Year label for the dataset.

    Returns
    -------
    pandas.DataFrame
    """
    df = df.copy()

    df = remove_prefixes(df)
    df = standardize_strings(df)

    # Add year label (cleaned later)
    df["year"] = f"year_{year}"

    # Drop columns that are completely empty
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        df = df.drop(columns=empty_cols)

    # Deduplicate column names if needed
    if not df.columns.is_unique:
        df.columns = deduplicate_columns(df.columns)

    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================
# 6. MERGE MULTIPLE DATASETS
# ============================================================

def merge_datasets(datasets):
    """
    Merge all cleaned datasets into one DataFrame.

    Parameters
    ----------
    datasets : dict
        {
            2008: df_2008_cleaned,
            2009: df_2009_cleaned,
            ...
        }

    Returns
    -------
    pandas.DataFrame
        Unified panel dataset with a cleaned integer year column.
    """
    merged = pd.concat(datasets.values(), ignore_index=True).copy()
    merged = clean_year_column(merged, "year")
    return merged
