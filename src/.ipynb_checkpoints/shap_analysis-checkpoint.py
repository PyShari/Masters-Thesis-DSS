import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# -------------------------
# Feature name mapping
# -------------------------
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
    "num__404": "Occupation (current job)",
    "num__402": "Sector",
    "num__122": "Retirement feeling",
    "num__289": "Expected retirement age",
    "num__008": "Education completed (2nd degree)",
    "num__479": "Savings deposit (2007)",
    "num__384": "Max full-time wage estimate",
    "num__136": "Commute to work (minutes)",
    "num__032": "Education–work match scale",
    "num__003": "Age",
    "num__022": "Preferred retirement age",
    "num__459": "Lifecourse deposit (2007)",
    "num__438": "No children or grandchildren",
    "num__450": "Provides informal care",
    "num__123": "Organisation type (first job)",
    "num__035": "Has taken job courses",
    "binary__437": "Has grandchildren",
    "binary__492": "Reason <36h: family/health",
    "num__missingindicator_141": "Missing: weekend work frequency",
    "num__missingindicator_142": "Missing: evening work frequency",
    "num__missingindicator_517": "Missing: minimum wage offer",
    "num__missingindicator_405": "Missing: occupation (first job)",
    "num__missingindicator": "Missing: irregular work hours",
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
    "num__141": "Weekend work frequency",
    "binary__395": "Reason <36h: health"
}

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

def safe_save_plot(fig, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)
    fig.savefig(full_path, bbox_inches='tight')
    plt.close(fig)
    return full_path

def compute_shap_xgb_per_regime(results, regime_name, model_name="XGBoost", output_dir="shap_plots", feature_name_map=None):
    """
    Compute SHAP values for a single regime using the full test set.
    Displays and saves the top 15 feature beeswarm plot and prints mean absolute SHAP values.
    Supports human-readable feature names.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract model and data
    res = results[model_name]
    pipeline = res["pipeline"]
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    X_test = res["X_test"]
    feature_names = preprocessor.get_feature_names_out()


    # Use the full test set
    X_full = X_test.copy()
    X_transformed = preprocessor.transform(X_full)
    X_df = pd.DataFrame(X_transformed, columns=feature_names)


    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    # Mean absolute SHAP
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values(ascending=False)
    top_features = mean_abs_shap.head(15).index.tolist()

    # Human-readable names for top features
    top_features_display = [feature_name_map.get(f, f) for f in top_features] if feature_name_map else top_features
    mean_abs_shap_display = mean_abs_shap[top_features].rename(index=lambda x: feature_name_map.get(x, x) if feature_name_map else x)

    print("\nMean Absolute SHAP Values (Top 15 Features):")
    display(mean_abs_shap_display)

    # Beeswarm plot
    plt.figure(figsize=(10,6))
    shap.summary_plot(
        shap_values,
        X_df,
        feature_names=[feature_name_map.get(f, f) if feature_name_map else f for f in feature_names],
        max_display=15,
        plot_type="dot",
        show=False
    )
    plt.title(f"SHAP Beeswarm - {regime_name}")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"shap_beeswarm_{regime_name}.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Beeswarm saved to: {save_path}")

    # Return all necessary info
    return {
        "mean_abs_shap": mean_abs_shap,
        "shap_values": shap_values,
        "X_full": pd.DataFrame(X_transformed, columns=feature_names)  # aligned with SHAP values
    }


def compute_across_regime_shap_common_features(
    shap_results_per_regime,
    top_n_features=15,
    output_dir="shap_across_regimes",
    feature_name_map=None
):
    """
    Compute SHAP analyses across regimes using only raw-name common features.
    Readable names are used ONLY for display, never for indexing.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1 — Find TRUE common raw features across regimes
    # ---------------------------------------------------------
    feature_sets = [
        set(res["mean_abs_shap"].index)
        for res in shap_results_per_regime.values()
    ]

    common_features = sorted(set.intersection(*feature_sets))

    print(f"\nCommon Features Across Regimes ({len(common_features)} raw features):")
    display(common_features)

    # Readable display version (ONLY for printing)
    if feature_name_map:
        common_features_display = [feature_name_map.get(f, f) for f in common_features]
        print("\nCommon Features (Readable Names):")
        display(common_features_display)
    else:
        common_features_display = common_features

    # ---------------------------------------------------------
    # STEP 2 — Build mean_abs_shap DataFrame for common features
    # ---------------------------------------------------------
    mean_abs_shap_df = pd.DataFrame({
        regime: res["mean_abs_shap"].loc[common_features]
        for regime, res in shap_results_per_regime.items()
    })

    # Display readable version
    if feature_name_map:
        print("\nMean Absolute SHAP Values for Common Features (Readable Names):")
        display(mean_abs_shap_df.rename(index=feature_name_map))
    else:
        print("\nMean Absolute SHAP Values for Common Features:")
        display(mean_abs_shap_df)

    # ---------------------------------------------------------
    # STEP 3 — Rank features and compute variance
    # ---------------------------------------------------------
    rank_df = mean_abs_shap_df.rank(ascending=False)
    spearman_matrix = rank_df.corr(method="spearman")
    rank_variance = rank_df.var(axis=1)

    print("\nSpearman Correlation of Feature Ranks Across Regimes:")
    display(spearman_matrix)

    print("\nRank Variance Across Regimes:")
    if feature_name_map:
        display(rank_variance.rename(index=feature_name_map).sort_values(ascending=False).to_frame("Rank Variance"))
    else:
        display(rank_variance.sort_values(ascending=False).to_frame("Rank Variance"))

    # ---------------------------------------------------------
    # STEP 4 — Select top features by mean SHAP across regimes
    # ---------------------------------------------------------
    mean_abs_shap_df["mean_across_regimes"] = mean_abs_shap_df.mean(axis=1)
    top_common_features = (
        mean_abs_shap_df["mean_across_regimes"]
        .sort_values(ascending=False)
        .head(top_n_features)
        .index
    )

    # Readable names for plotting
    top_common_features_display = (
        [feature_name_map.get(f, f) for f in top_common_features]
        if feature_name_map else top_common_features
    )

    # ---------------------------------------------------------
    # STEP 5 — Beeswarm plots per regime (raw names for indexing)
    # ---------------------------------------------------------
    for regime, res in shap_results_per_regime.items():
        print(f"\nBeeswarm for regime: {regime}")

        # Raw-name indexing (SAFE)
        feat_indices = [res["mean_abs_shap"].index.get_loc(f) for f in top_common_features]

        shap_vals_subset = res["shap_values"][:, feat_indices]
        X_subset = res["X_full"].iloc[:, feat_indices]

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_vals_subset,
            X_subset,
            feature_names=top_common_features_display,
            max_display=top_n_features,
            plot_type="dot",
            show=False
        )
        plt.title(f"SHAP Beeswarm - {regime}")
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"shap_beeswarm_{regime}.png")
        plt.savefig(save_path)
        plt.show()
        print(f"Saved beeswarm to: {save_path}")

    # ---------------------------------------------------------
    # RETURN raw names + readable names (separate)
    # ---------------------------------------------------------
    return {
        "common_features_raw": common_features,
        "common_features_readable": common_features_display,
        "mean_abs_shap_df": mean_abs_shap_df.drop(columns="mean_across_regimes"),
        "rank_df": rank_df,
        "spearman_matrix": spearman_matrix,
        "rank_variance_raw": rank_variance,
        "rank_variance_readable": (
            rank_variance.rename(index=feature_name_map)
            if feature_name_map else rank_variance
        )
    }

def summarize_volatile_stable_features(across_regime_results, top_n=10, feature_name_map=None):
    """
    Identify top volatile (high rank variance) and stable (low rank variance) features across regimes.
    Uses human-readable names ONLY for display.
    Raw names are preserved internally.
    """

    # Extract raw rank variance and SHAP table
    rank_variance_raw = across_regime_results["rank_variance_raw"]
    mean_abs_shap_df = across_regime_results["mean_abs_shap_df"]

    # Identify volatile and stable features (raw names)
    volatile_features_raw = rank_variance_raw.sort_values(ascending=False).head(top_n).index
    stable_features_raw   = rank_variance_raw.sort_values(ascending=True).head(top_n).index

    # Build tables (raw names)
    volatile_table = mean_abs_shap_df.loc[volatile_features_raw].copy()
    volatile_table["Rank Variance"] = rank_variance_raw.loc[volatile_features_raw]

    stable_table = mean_abs_shap_df.loc[stable_features_raw].copy()
    stable_table["Rank Variance"] = rank_variance_raw.loc[stable_features_raw]

    # Display readable names if mapping provided
    if feature_name_map:
        volatile_table_display = volatile_table.rename(index=feature_name_map)
        stable_table_display   = stable_table.rename(index=feature_name_map)
    else:
        volatile_table_display = volatile_table
        stable_table_display   = stable_table

    print("\nTop Volatile Features Across Regimes (High Rank Variance):")
    display(volatile_table_display)

    print("\nTop Stable Features Across Regimes (Low Rank Variance):")
    display(stable_table_display)

    return {
        "volatile_features_raw": volatile_features_raw,
        "stable_features_raw": stable_features_raw,
        "volatile_table": volatile_table,
        "stable_table": stable_table,
        "volatile_table_readable": volatile_table_display,
        "stable_table_readable": stable_table_display
    }
