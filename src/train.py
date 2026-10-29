"""
train.py - Leakage-Safe Cross-Validation, Model Selection, and Threshold Optimization for CrediSHAP.

Key Principles:
1. Imputation, encoding, and SMOTE are combined into an imbalanced-learn Pipeline
   and fitted independently inside every CV fold.
2. Candidate models: Random Forest, XGBoost, LightGBM.
3. Model Selection Rule: Highest 3-fold Stratified CV F1-score on the original distribution.
4. Out-of-fold (OOF) probability predictions are generated strictly on un-resampled validation partitions.
5. Decision Threshold Optimization: Maximizes F1-score across OOF predictions without test set leakage.
6. Deployment Retraining: Final candidate pipelines are fitted on the complete training set (118,936 samples)
   with SMOTE applied exclusively within the pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from src.preprocessing import build_pipeline


def train_and_compare_models(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    tuning_results: dict,
    cv_folds: int = 3,
    random_state: int = 42
):
    """
    Evaluates candidate models via 3-Fold Stratified Cross-Validation using
    leakage-safe imbalanced-learn pipelines fitted independently inside every fold.
    Selects champion model by highest CV F1-score and optimizes decision threshold on OOF predictions.
    """
    print("\n--- Cross-Validation Model Training & Comparison ---")
    print("    (Independent per-fold imputation, encoding, and SMOTE; validation on original distribution)")

    rf_params = tuning_results.get("Random Forest", {}).get("best_params", {
        'n_estimators': 100, 'max_depth': 12, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'
    })
    xgb_params = tuning_results.get("XGBoost", {}).get("best_params", {
        'n_estimators': 120, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'tree_method': 'hist', 'eval_metric': 'logloss'
    })
    lgb_params = tuning_results.get("LightGBM", {}).get("best_params", {
        'n_estimators': 120, 'max_depth': 6, 'num_leaves': 31, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'verbose': -1
    })

    models = {
        "Random Forest": RandomForestClassifier(**rf_params, random_state=random_state, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(**xgb_params, random_state=random_state, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(**lgb_params, random_state=random_state, n_jobs=-1)
    }

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results = {}
    oof_predictions = {}

    for name, clf in models.items():
        print(f"Evaluating {name}...")
        oof_probs = np.zeros(len(y_train), dtype=float)
        fold_metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1": [], "ROC-AUC": [], "PR-AUC": []}

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_df, y_train), 1):
            X_fold_train = X_train_df.iloc[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train_df.iloc[val_idx]
            y_fold_val = y_train[val_idx]

            # Fit leakage-safe pipeline independently on fold-train
            pipe = build_pipeline(clf, random_state=random_state)
            pipe.fit(X_fold_train, y_fold_train)

            # Predict probabilities on un-resampled validation fold
            probs = pipe.predict_proba(X_fold_val)[:, 1]
            oof_probs[val_idx] = probs
            preds = (probs >= 0.50).astype(int)

            fold_metrics["Accuracy"].append(accuracy_score(y_fold_val, preds))
            fold_metrics["Precision"].append(precision_score(y_fold_val, preds, zero_division=0))
            fold_metrics["Recall"].append(recall_score(y_fold_val, preds, zero_division=0))
            fold_metrics["F1"].append(f1_score(y_fold_val, preds, zero_division=0))
            fold_metrics["ROC-AUC"].append(roc_auc_score(y_fold_val, probs))
            fold_metrics["PR-AUC"].append(average_precision_score(y_fold_val, probs))

        cv_results[name] = {metric: float(np.mean(vals)) for metric, vals in fold_metrics.items()}
        oof_predictions[name] = oof_probs

        print(f"  {name} CV Metrics -> ROC-AUC: {cv_results[name]['ROC-AUC']:.4f}, "
              f"PR-AUC: {cv_results[name]['PR-AUC']:.4f}, "
              f"F1: {cv_results[name]['F1']:.4f}, "
              f"Recall: {cv_results[name]['Recall']:.4f}, "
              f"Precision: {cv_results[name]['Precision']:.4f}")

    # Consistent Model Selection Rule: Highest CV F1-score
    best_model_name = max(cv_results, key=lambda m: cv_results[m]["F1"])
    print(f"\nChampion Model Selected: '{best_model_name}' (Highest CV F1-score: {cv_results[best_model_name]['F1']:.4f})")

    # Optimal Decision Threshold Selection on Champion OOF predictions
    best_oof_probs = oof_predictions[best_model_name]
    candidate_thresholds = np.arange(0.10, 0.91, 0.01)
    best_thresh = 0.50
    best_thresh_f1 = -1.0

    for t in candidate_thresholds:
        t_preds = (best_oof_probs >= t).astype(int)
        score = f1_score(y_train, t_preds, zero_division=0)
        if score > best_thresh_f1:
            best_thresh_f1 = score
            best_thresh = float(np.round(t, 2))

    print(f"Optimal Decision Threshold Selected via OOF CV: {best_thresh:.2f} (Locked prior to test evaluation)")

    # Retrain final deployment pipelines on complete training set (118,936 samples)
    print("\nRetraining candidate pipelines on complete training set for deployment...")
    final_pipelines = {}
    for name, clf in models.items():
        print(f"  Fitting final {name} pipeline on full training data...")
        pipe = build_pipeline(clf, random_state=random_state)
        pipe.fit(X_train_df, y_train)
        final_pipelines[name] = pipe

    # Count final SMOTE balanced samples
    n_class_0 = int(np.sum(y_train == 0))
    smote_total = n_class_0 * 2  # 50:50 parity
    smote_info = {
        "original_train_samples": len(y_train),
        "balanced_train_samples": smote_total,
        "class_0_samples": n_class_0,
        "class_1_samples": n_class_0
    }

    return {
        "best_model_name": best_model_name,
        "optimal_threshold": best_thresh,
        "cv_results": cv_results,
        "oof_predictions": oof_predictions,
        "final_pipelines": final_pipelines,
        "smote_info": smote_info
    }
