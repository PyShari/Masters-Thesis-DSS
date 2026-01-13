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

def compute_all_subgroup_profiles(shap_df, subgroup_cols):
    """
    Compute mean |SHAP| profiles for all subgroup variables.
    """

    feature_cols = shap_df.columns.difference(subgroup_cols)

    subgroup_results = {}

    for subgroup in subgroup_cols:
        profiles = (
            shap_df
            .groupby(subgroup)[feature_cols]
            .apply(lambda x: x.abs().mean())
        )

        subgroup_results[subgroup] = profiles

        print(f"\nMean |SHAP| by {subgroup}:")
        display(profiles)
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

    return subgroup_results

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
