# shap_core.py
import shap
import numpy as np
from sklearn.cluster import KMeans

def build_shap_background(pipeline, X_train, n_clusters=100):
    X_trans = pipeline.named_steps["preprocessor"].transform(X_train)
    return KMeans(n_clusters=n_clusters, random_state=42).fit(X_trans).cluster_centers_

def compute_tree_shap(pipeline, X_train, X_test, background):
    X_test_trans = pipeline.named_steps["preprocessor"].transform(X_test)
    explainer = shap.TreeExplainer(
        pipeline.named_steps["model"],
        data=background
    )
    return explainer.shap_values(X_test_trans), X_test_trans
