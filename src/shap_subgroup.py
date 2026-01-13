import numpy as np
import pandas as pd

def build_subgroups(df):
    """
    Build all subgroup labels on the RAW test dataframe.
    Returns df with new subgroup columns.
    """

    df = df.copy()

    # -----------------------------
    # Age group
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
            1: "Agriculture & Extraction",
            2: "Agriculture & Extraction",
            3: "Production & Construction",
            4: "Production & Construction",
            5: "Production & Construction",
            6: "Trade & Market Services",
            7: "Trade & Market Services",
            8: "Trade & Market Services",
            9: "Trade & Market Services",
            10: "Trade & Market Services",
            11: "Public & Social Services",
            12: "Public & Social Services",
            13: "Public & Social Services",
            14: "Culture & Other Services",
            15: "Culture & Other Services"
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

def attach_all_subgroups(shap_results):
    """
    Attach all subgroup labels to SHAP dataframe using index alignment.
    """

    shap_df = shap_results["shap_df"].copy()
    df_test = shap_results["df_test"]

    df_with_groups = build_subgroups(df_test)

    subgroup_cols = [
        c for c in df_with_groups.columns
        if c.endswith('_group') or c in ['life_situation']
    ]

    for col in subgroup_cols:
        shap_df[col] = df_with_groups[col].loc[shap_df.index]

    return shap_df, subgroup_cols

def subgroup_rank_variance(subgroup_profiles, top_n=15):
    """
    Compute rank variance across subgroup categories for each subgroup variable.
    """

    rank_variance_results = {}

    for subgroup, df in subgroup_profiles.items():
        rank_df = df.rank(axis=1, ascending=False)
        rank_variance = rank_df.var(axis=0).sort_values(ascending=False)

        rank_variance_results[subgroup] = rank_variance

        print(f"\nTop volatile features across {subgroup}:")
        display(rank_variance.head(top_n).to_frame("Rank Variance"))

    return rank_variance_results

import matplotlib.pyplot as plt


def plot_all_subgroup_profiles(subgroup_profiles, top_n=15):
    """
    Plot SHAP profile comparisons for all subgroup variables.
    """

    for subgroup, df in subgroup_profiles.items():
        top_features = (
            df.mean(axis=0)
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )

        df[top_features].T.plot(
            kind="bar",
            figsize=(12, 6)
        )

        plt.ylabel("Mean |SHAP|")
        plt.title(f"Subgroup SHAP Profiles by {subgroup}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def compute_subgroup_shap_analysis(
    shap_values,
    X_test,
    df_regime_raw,
    feature_names,
    build_subgroups_fn,
    top_n=20,
    cmap="coolwarm"
):
    """
    Full pipeline:
    1. Align raw test rows with SHAP rows
    2. Build subgroup labels
    3. Attach subgroup labels to SHAP dataframe
    4. Compute subgroup SHAP profiles
    5. Compute subgroup-level rank variance
    6. Generate heatmaps comparing subgroups
    """

    # ---------------------------------------------------------
    # STEP 1 — Align raw test rows with SHAP rows
    # ---------------------------------------------------------
    df_test = df_regime_raw.loc[X_test.index].copy()

    # SHAP dataframe
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    # ---------------------------------------------------------
    # STEP 2 — Build subgroup labels on raw test data
    # ---------------------------------------------------------
    df_test_groups = build_subgroups_fn(df_test)

    # Identify subgroup columns
    subgroup_cols = [
        c for c in df_test_groups.columns
        if c.endswith("_group") or c in ["life_situation"]
    ]

    # ---------------------------------------------------------
    # STEP 3 — Attach subgroup labels to SHAP dataframe
    # ---------------------------------------------------------
    for col in subgroup_cols:
        shap_df[col] = df_test_groups[col].values

    # ---------------------------------------------------------
    # STEP 4 — Compute subgroup SHAP profiles
    # ---------------------------------------------------------
    subgroup_profiles = {}

    for subgroup in subgroup_cols:
        profile = (
            shap_df.groupby(subgroup)[feature_names]
            .mean()
            .T  # features as rows
        )
        subgroup_profiles[subgroup] = profile

    # ---------------------------------------------------------
    # STEP 5 — Compute subgroup-level rank variance
    # ---------------------------------------------------------
    subgroup_rank_variance = {}

    for subgroup, profile in subgroup_profiles.items():
        rank_df = profile.rank(ascending=False, axis=0)
        rank_var = rank_df.var(axis=1).sort_values(ascending=False)
        subgroup_rank_variance[subgroup] = rank_var

    # ---------------------------------------------------------
    # STEP 6 — Generate heatmaps comparing subgroups
    # ---------------------------------------------------------
    for subgroup, profile in subgroup_profiles.items():

        # Select top features for visualization
        top_features = (
            profile.mean(axis=1)
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            profile.loc[top_features],
            cmap=cmap,
            annot=False,
            linewidths=0.5
        )
        plt.title(f"Subgroup SHAP Profiles — {subgroup}")
        plt.xlabel("Subgroup Category")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------
    # RETURN EVERYTHING
    # ---------------------------------------------------------
    return {
        "shap_df": shap_df,
        "subgroup_cols": subgroup_cols,
        "subgroup_profiles": subgroup_profiles,
        "subgroup_rank_variance": subgroup_rank_variance
    }

def rename_features(df, feature_name_map):
    return df.rename(index=feature_name_map)
