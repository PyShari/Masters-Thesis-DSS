"""
plotting.py
-----------

This module provides visualization utilities for evaluating model
performance across economic regimes. It includes:

1. Error distribution plots (KDE)
2. RMSE per fold / validation RMSE plots
3. Predicted vs Actual scatterplots

All plots use consistent regime‑specific color palettes and automatically
save high‑resolution PNG/PDF files for inclusion in the thesis.
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils import safe_filename


# ============================================================
# COLOR PALETTE FOR REGIMES
# ============================================================

REGIME_COLORS = {
    "Regime A": "#1f77b4",   # blue
    "Regime B": "#ff7f0e",   # orange
    "Regime C": "#2ca02c"    # green
}


# ============================================================
# 1. ERROR DISTRIBUTION PLOT
# ============================================================

def plot_error_distribution(results, regime_name):
    """
    Visualize the distribution of prediction errors for all models
    within a regime using kernel density estimation (KDE).

    Helps assess:
    - Systematic overprediction / underprediction
    - Differences in error spread between models
    - Whether errors are centered around zero (ideal)

    Parameters
    ----------
    results : dict
        {
            "ModelName": {
                "errors": np.array([...])
            },
            ...
        }
    regime_name : str
        Full regime label, e.g. "Regime A (2008–2013)".
    """
    plt.figure(figsize=(10, 5))

    # Extract "Regime A" from "Regime A (2008–2013)"
    regime_key = regime_name.split("(")[0].strip()
    color = REGIME_COLORS.get(regime_key, "#333333")

    # Plot KDE for each model
    for model_name, res in results.items():
        sns.kdeplot(
            res["errors"],
            label=model_name,
            fill=True,
            alpha=0.3,
            color=color
        )

    # Reference line at zero
    plt.axvline(0, color="black", linestyle="--")

    plt.title(f"Prediction Error Distribution – {regime_name}")
    plt.xlabel("Prediction Error")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    # Save
    safe_regime = safe_filename(regime_key)
    plt.savefig(f"error_distribution_{safe_regime}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"error_distribution_{safe_regime}.pdf", dpi=300, bbox_inches="tight")

    plt.show()


# ============================================================
# 2. RMSE PER FOLD / VALIDATION RMSE
# ============================================================

def plot_per_fold_rmse(results, regime_name):
    """
    Plot RMSE values across CV folds or validation RMSE for each model.

    Helps identify:
    - Stability across folds
    - Overfitting (large variation)
    - Whether validation RMSE aligns with test performance

    Parameters
    ----------
    results : dict
        {
            "ModelName": {
                "per_fold_rmse": [...],   # optional
                "val_rmse": float         # optional
            }
        }
    regime_name : str
        Full regime label.
    """
    plt.figure(figsize=(10, 5))

    regime_key = regime_name.split("(")[0].strip()
    color = REGIME_COLORS.get(regime_key, "#333333")

    for model_name, res in results.items():

        # Cross‑validation RMSE curve
        if "per_fold_rmse" in res:
            plt.plot(
                res["per_fold_rmse"],
                marker="o",
                label=f"{model_name} (CV)",
                color=color
            )

        # Single validation RMSE
        elif "val_rmse" in res and res["val_rmse"] is not None:
            plt.axhline(
                res["val_rmse"],
                linestyle="--",
                label=f"{model_name} (Validation)",
                color=color
            )

    plt.title(f"Validation / CV RMSE – {regime_name}")
    plt.xlabel("Fold (if applicable)")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_regime = safe_filename(regime_key)
    plt.savefig(f"rmse_per_fold_{safe_regime}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"rmse_per_fold_{safe_regime}.pdf", dpi=300, bbox_inches="tight")

    plt.show()


# ============================================================
# 3. PREDICTED VS ACTUAL SCATTERPLOT
# ============================================================

def plot_predicted_vs_actual(model_name, y_true, y_pred, regime_name):
    """
    Scatterplot comparing predicted vs actual values for a model.

    Automatically saves the figure to a figures/ directory.

    Parameters
    ----------
    model_name : str
        Name of the model.
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Model predictions.
    regime_name : str
        Full regime label.
    """
    # Create folder
    save_dir = "figures"
    os.makedirs(save_dir, exist_ok=True)

    # Clean names for filenames
    regime_key = regime_name.split("(")[0].strip()
    safe_regime = safe_filename(regime_key)
    safe_model = safe_filename(model_name)

    save_path = os.path.join(save_dir, f"{safe_regime}_{safe_model}_pred_vs_actual.png")

    plt.figure(figsize=(6, 6))

    # Use consistent regime color
    color = REGIME_COLORS.get(regime_key, "#333333")

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.4,
        color=color,
        edgecolor="white",
        linewidth=0.5
    )

    # 45-degree reference line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)

    plt.title(f"{model_name}: Predicted vs Actual – {regime_name}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved plot to: {save_path}")
