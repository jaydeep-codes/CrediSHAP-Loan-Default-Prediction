"""
explain.py - SHAP Explainability Module for CrediSHAP.

Generates:
1. Global SHAP Feature Importance Bar Plot -> outputs/shap_summary.png
2. Global SHAP Directional Beeswarm Plot   -> outputs/shap_beeswarm.png
3. Local SHAP Waterfall Plot for Applicant -> outputs/local_explanation.png
4. Top Feature Importance rankings (mean absolute SHAP)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def compute_shap_explanations(
    model,
    X_sample: np.ndarray,
    feature_names: list,
    output_dir: str,
    applicant_index: int = 0
):
    """
    Computes TreeExplainer SHAP values on representative sample and generates plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Computing SHAP TreeExplainer on {len(X_sample)} sample instances ---")
    
    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(X_sample)
    
    # Handle binary classification shapes
    if len(shap_explanation.shape) == 3 and shap_explanation.shape[2] == 2:
        # If output has 2 classes [N, features, 2], select class 1 (default)
        shap_values_class1 = shap_explanation.values[:, :, 1]
        base_val = shap_explanation.base_values[:, 1]
    elif len(shap_explanation.shape) == 2:
        shap_values_class1 = shap_explanation.values
        base_val = shap_explanation.base_values
    else:
        shap_values_class1 = shap_explanation.values
        base_val = shap_explanation.base_values

    # Clean feature names for plotting
    clean_feature_names = [f.replace("[", "_").replace("]", "_").replace("<", "lt_").replace(">", "gt_") for f in feature_names]

    # Calculate mean absolute SHAP for ranking
    mean_abs_shap = np.mean(np.abs(shap_values_class1), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1]
    
    ranking = []
    for rank, idx in enumerate(top_indices[:15], 1):
        ranking.append({
            "rank": rank,
            "feature": feature_names[idx],
            "clean_feature": clean_feature_names[idx],
            "mean_abs_shap": float(mean_abs_shap[idx])
        })
    
    print("\n--- Top 10 Features by Mean Absolute SHAP Value ---")
    for item in ranking[:10]:
        print(f"  {item['rank']:2d}. {item['feature']:35} : {item['mean_abs_shap']:.4f}")

    # 1. Global Bar Plot
    bar_path = os.path.join(output_dir, "shap_summary.png")
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 7))
    shap.summary_plot(
        shap_values_class1,
        X_sample,
        feature_names=clean_feature_names,
        plot_type="bar",
        max_display=12,
        show=False
    )
    plt.title("Global Feature Importance (Mean |SHAP Value|)", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"Saved: {bar_path}")

    # 2. Beeswarm Plot
    beeswarm_path = os.path.join(output_dir, "shap_beeswarm.png")
    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 8))
    # Create Explanation object for beeswarm
    exp_obj = shap.Explanation(
        values=shap_values_class1,
        base_values=base_val,
        data=X_sample,
        feature_names=clean_feature_names
    )
    shap.plots.beeswarm(exp_obj, max_display=12, show=False)
    plt.title("SHAP Beeswarm Plot: Feature Impact Direction on Default Probability", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=300)
    plt.close()
    print(f"Saved: {beeswarm_path}")

    # 3. Local Waterfall Plot for Applicant
    local_path = os.path.join(output_dir, "local_explanation.png")
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 6))
    single_exp = exp_obj[applicant_index]
    shap.plots.waterfall(single_exp, max_display=10, show=False)
    plt.title(f"Local Attribution Waterfall Plot (Sample Applicant #{applicant_index})", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(local_path, dpi=300)
    plt.close()
    print(f"Saved: {local_path}")

    return {
        "ranking": ranking,
        "explainer": explainer,
        "clean_feature_names": clean_feature_names
    }
