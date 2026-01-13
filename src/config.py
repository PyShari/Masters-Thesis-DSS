"""
GLOBAL CONFIGURATION
--------------------

This dictionary centralizes all key hyperparameters and global settings
used throughout the modeling pipeline. Storing these values in a single
location ensures consistency across notebooks and scripts, improves
reproducibility, and makes it easy to adjust parameters without modifying
multiple files.

These settings control:
- target variable selection,
- randomization behavior,
- train/validation/test splits,
- cross‑validation behavior,
- target transformation,
- SHAP background sampling.
"""

GLOBAL_CONFIG = {
    # ---------------------------------------------------------
    # Target variable
    # ---------------------------------------------------------
    # Column "127" corresponds to weekly working hours in the LISS dataset.
    # All models will predict this variable unless overridden.
    "y_col": "127",

    # ---------------------------------------------------------
    # Randomization control
    # ---------------------------------------------------------
    # Ensures reproducibility across all models, splits, and SHAP sampling.
    "random_state": 42,

    # ---------------------------------------------------------
    # Train/validation/test split proportions
    # ---------------------------------------------------------
    # 15% of the data is held out for testing (final evaluation).
    "test_size": 0.15,

    # 15% of the remaining training data is used for validation
    # (hyperparameter tuning and early stopping).
    "val_size": 0.15,

    # ---------------------------------------------------------
    # Cross‑validation settings
    # ---------------------------------------------------------
    # Number of folds used for model selection and hyperparameter tuning.
    "cv_splits": 5,

    # ---------------------------------------------------------
    # Target transformation
    # ---------------------------------------------------------
    # If True, the target variable is log‑transformed before modeling.
    # This is useful when the target distribution is right‑skewed
    # (common for working hours due to part‑time workers).
    "log_target": False,

    # ---------------------------------------------------------
    # SHAP background sampling
    # ---------------------------------------------------------
    # Number of k‑means clusters used to summarize the background dataset
    # for SHAP KernelExplainer or TreeExplainer.
    # Reduces computational cost while preserving representative structure.
    "shap_background_clusters": 10
}
