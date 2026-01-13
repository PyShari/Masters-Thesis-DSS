"""
data_loading.py
----------------

This module provides utilities for loading raw LISS panel datasets
from .dta (Stata) files. The primary function, `load_liss_datasets`,
reads multiple yearly files into a dictionary of pandas DataFrames,
ensuring consistent structure and robust error handling.

This module is imported by the main cleaning pipeline and notebooks.
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import pandas as pd


# ============================================================
# MAIN FUNCTION: LOAD MULTIPLE LISS DATASETS
# ============================================================

def load_liss_datasets(folder_path, file_year_map):
    """
    Load multiple LISS .dta files into a dictionary of DataFrames.

    Parameters
    ----------
    folder_path : str
        Path to the directory containing all .dta files.
        Example:
            "C:/Users/.../Thesis_DSS_2026/data"

    file_year_map : dict
        Mapping of survey year → filename.
        Example:
            {
                2008: "cw08a_EN_1.1p.dta",
                2009: "cw09b_EN_3.0p.dta",
                ...
            }

    Returns
    -------
    datasets : dict[int, pandas.DataFrame]
        Dictionary where keys are years and values are loaded DataFrames.

    Notes
    -----
    - Missing files are skipped with a warning instead of raising an error.
    - All DataFrames are loaded with `convert_categoricals=False` to avoid
      Stata category dtype issues.
    - Ensures reproducibility and consistent loading across all years.
    """
    datasets = {}

    for year, filename in file_year_map.items():
        file_path = os.path.join(folder_path, filename)

        if not os.path.exists(file_path):
            print(f"[WARNING] File not found for year {year}: {filename}")
            continue

        try:
            df = pd.read_stata(file_path, convert_categoricals=False)
            datasets[year] = df
            print(f"[OK] Loaded {filename} for year {year} (rows={len(df)})")

        except Exception as e:
            print(f"[ERROR] Failed to load {filename} for year {year}: {e}")

    return datasets
