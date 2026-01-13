# shap_aggregation.py
import numpy as np

def build_raw_to_transformed_map(feature_names, kept_features):
    feature_names = np.array(feature_names, dtype=str)
    mapping = {}

    for raw in kept_features:
        idx = np.where(np.char.startswith(feature_names, raw))[0]
        if len(idx) > 0:
            mapping[raw] = idx.tolist()

    if len(mapping) == 0:
        raise ValueError("No raw features map to transformed features.")

    return mapping


def aggregate_shap_by_raw_feature(shap_values, raw_to_transformed):
    """
    Returns:
        dict: raw_feature -> aggregated SHAP vector (n_samples,)
    """
    agg = {}
    for raw, idxs in raw_to_transformed.items():
        agg[raw] = shap_values[:, idxs].sum(axis=1)
    return agg


def mean_abs_shap(agg_shap):
    return {
        raw: np.mean(np.abs(vals))
        for raw, vals in agg_shap.items()
    }
