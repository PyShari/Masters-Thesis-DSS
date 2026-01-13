# shap_plots.py
import numpy as np
import shap
import matplotlib.pyplot as plt

def plot_raw_beeswarm(
    agg_shap,
    X_raw,
    feature_name_map,
    top_n=15,
    title=None,
    save_path=None
):
    mean_abs = {
        k: np.mean(np.abs(v))
        for k, v in agg_shap.items()
    }

    top_features = sorted(mean_abs, key=mean_abs.get, reverse=True)[:top_n]

    X_plot = np.column_stack([X_raw[f].values for f in top_features])
    shap_plot = np.column_stack([agg_shap[f] for f in top_features])

    feature_labels = [feature_name_map.get(f, f) for f in top_features]

    plt.figure(figsize=(14, 10))
    shap.summary_plot(
        shap_plot,
        X_plot,
        feature_names=feature_labels,
        show=False
    )

    if title:
        plt.title(title)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)

    plt.show()
