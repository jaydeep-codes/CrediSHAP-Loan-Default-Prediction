"""
evaluate.py - Final Evaluation and Visualization Module for CrediSHAP.

Evaluates all models on the isolated test set (29,734 samples) ONCE.
Generates research-grade figures:
1. outputs/confusion_matrix.png
2. outputs/roc_curve.png
3. outputs/precision_recall_curve.png
4. outputs/model_comparison.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report
)

# Style settings for research-grade figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11


def evaluate_model_on_test(model, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5):
    """Computes comprehensive metrics on test data at given threshold."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    metrics = {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "ROC-AUC": float(roc_auc),
        "PR-AUC": float(pr_auc),
        "Specificity": float(specificity),
        "FNR": float(fnr),
        "FPR": float(fpr),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Confusion_Matrix": cm.tolist(),
        "Threshold": float(threshold)
    }
    return metrics, probs, preds


def evaluate_all_models_and_plot(
    trained_models: dict,
    champion_name: str,
    optimal_threshold: float,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str
):
    """
    Evaluates all candidate models on untouched test set, generates publication-quality figures,
    and returns complete test results dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_test_metrics = {}
    model_probs = {}
    
    print(f"\n--- Final Test Set Evaluation ({len(y_test):,} untouched samples) ---")
    for name, model in trained_models.items():
        # Evaluate at default 0.50
        metrics_05, probs, _ = evaluate_model_on_test(model, X_test, y_test, threshold=0.50)
        all_test_metrics[name] = metrics_05
        model_probs[name] = probs
        print(f"{name} (threshold 0.50): ROC-AUC={metrics_05['ROC-AUC']:.4f}, PR-AUC={metrics_05['PR-AUC']:.4f}, Recall={metrics_05['Recall']:.4f}, Precision={metrics_05['Precision']:.4f}, F1={metrics_05['F1']:.4f}, Acc={metrics_05['Accuracy']:.4f}")
    
    # Champion at optimal threshold
    champ_model = trained_models[champion_name]
    champ_opt_metrics, champ_opt_probs, champ_opt_preds = evaluate_model_on_test(
        champ_model, X_test, y_test, threshold=optimal_threshold
    )
    all_test_metrics[f"{champion_name} (Optimal Thresh {optimal_threshold})"] = champ_opt_metrics
    print(f"\nChampion '{champion_name}' at Optimal Threshold ({optimal_threshold}):")
    print(f"  Recall={champ_opt_metrics['Recall']:.4f}, Precision={champ_opt_metrics['Precision']:.4f}, F1={champ_opt_metrics['F1']:.4f}, Acc={champ_opt_metrics['Accuracy']:.4f}, ROC-AUC={champ_opt_metrics['ROC-AUC']:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, champ_opt_preds))

    # 1. Confusion Matrix Plot
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.array(champ_opt_metrics["Confusion_Matrix"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Non-Default (0)', 'Default (1)'],
                yticklabels=['Non-Default (0)', 'Default (1)'],
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"Confusion Matrix: {champion_name}\n(Test N={len(y_test)}, Threshold={optimal_threshold})", fontsize=12, pad=12)
    ax.set_ylabel("True Class", fontsize=11)
    ax.set_xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")

    # 2. ROC Curves Plot
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.figure(figsize=(7, 6))
    colors = {'Random Forest': '#2ca02c', 'XGBoost': '#ff7f0e', 'LightGBM': '#1f77b4'}
    for name, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = all_test_metrics[name]['ROC-AUC']
        lw = 2.5 if name == champion_name else 1.5
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=colors.get(name, 'black'), lw=lw)
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7, label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=12, pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved: {roc_path}")

    # 3. Precision-Recall Curves Plot
    pr_path = os.path.join(output_dir, "precision_recall_curve.png")
    plt.figure(figsize=(7, 6))
    for name, probs in model_probs.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = all_test_metrics[name]['PR-AUC']
        lw = 2.5 if name == champion_name else 1.5
        plt.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.4f})", color=colors.get(name, 'black'), lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.title("Precision-Recall (PR) Curve Comparison", fontsize=12, pad=12)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"Saved: {pr_path}")

    # 4. Model Comparison Bar Chart
    comp_path = os.path.join(output_dir, "model_comparison.png")
    comparison_df = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": all_test_metrics[name]["Accuracy"],
            "Precision": all_test_metrics[name]["Precision"],
            "Recall": all_test_metrics[name]["Recall"],
            "F1-Score": all_test_metrics[name]["F1"],
            "ROC-AUC": all_test_metrics[name]["ROC-AUC"],
            "PR-AUC": all_test_metrics[name]["PR-AUC"],
        }
        for name in trained_models.keys()
    ])
    
    comp_melted = pd.melt(comparison_df, id_vars=["Model"], var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=comp_melted, x="Metric", y="Score", hue="Model", palette=["#2ca02c", "#ff7f0e", "#1f77b4"])
    plt.ylim([0.0, 1.05])
    plt.title("Cross-Model Performance Comparison on Untouched Test Set", fontsize=12, pad=12)
    plt.ylabel("Metric Score", fontsize=11)
    plt.xlabel("")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Saved: {comp_path}")

    return all_test_metrics, comparison_df
