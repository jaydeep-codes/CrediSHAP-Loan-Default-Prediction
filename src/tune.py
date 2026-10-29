"""
tune.py - Bayesian Hyperparameter Optimization with Optuna for CrediSHAP.

Tunes:
1. Random Forest (scikit-learn)
2. XGBoost (xgboost)
3. LightGBM (lightgbm)

Uses Stratified K-Fold CV on the ORIGINAL (non-resampled) training data.
Every fold independently fits the complete imbalanced-learn Pipeline:
- Median imputation for numerical features
- Most-frequent imputation + One-Hot Encoding for categorical features
- SMOTE applied exclusively to the fold's training partition
- Validation partition evaluated on original class distribution without resampling

Objective: Maximize F1-score on the minority default class.
"""

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import f1_score

from src.preprocessing import build_pipeline

# Disable verbose optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _subsample_if_large(X: pd.DataFrame, y: np.ndarray, max_size: int = 30000, random_state: int = 42):
    """Subsample for tuning speed if training set is very large."""
    if len(X) > max_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), size=max_size, replace=False)
        return X.iloc[idx].reset_index(drop=True), y[idx]
    return X.reset_index(drop=True), y


def tune_random_forest(X_train_df: pd.DataFrame, y_train: np.ndarray, n_trials: int = 10, cv_folds: int = 3, random_state: int = 42):
    """
    Bayesian optimization for Random Forest.
    Uses leakage-safe imbalanced-learn pipeline fitted independently per fold.
    """
    print(f"\n[Optuna] Tuning Random Forest ({n_trials} trials, {cv_folds}-fold CV)...")
    X_sub, y_sub = _subsample_if_large(X_train_df, y_train, random_state=random_state)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150, step=25),
            'max_depth': trial.suggest_int('max_depth', 8, 16),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state': random_state,
            'n_jobs': -1
        }
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = []
        for train_idx, val_idx in skf.split(X_sub, y_sub):
            X_fold_train = X_sub.iloc[train_idx]
            y_fold_train = y_sub[train_idx]
            X_fold_val = X_sub.iloc[val_idx]
            y_fold_val = y_sub[val_idx]
            
            # Leakage-safe pipeline fitted independently on fold-train
            clf = RandomForestClassifier(**params)
            pipe = build_pipeline(clf, random_state=random_state)
            pipe.fit(X_fold_train, y_fold_train)
            
            val_probs = pipe.predict_proba(X_fold_val)[:, 1]
            val_preds = (val_probs >= 0.50).astype(int)
            scores.append(f1_score(y_fold_val, val_preds, zero_division=0))
        return np.mean(scores)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Random Forest Best Trial: F1={study.best_value:.4f}")
    print(f"Random Forest Best Params: {study.best_params}")
    return study.best_params, study.best_value


def tune_xgboost(X_train_df: pd.DataFrame, y_train: np.ndarray, n_trials: int = 10, cv_folds: int = 3, random_state: int = 42):
    """
    Bayesian optimization for XGBoost.
    Uses leakage-safe imbalanced-learn pipeline fitted independently per fold.
    """
    print(f"\n[Optuna] Tuning XGBoost ({n_trials} trials, {cv_folds}-fold CV)...")
    X_sub, y_sub = _subsample_if_large(X_train_df, y_train, random_state=random_state)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 80, 200, step=20),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'tree_method': 'hist',
            'eval_metric': 'logloss',
            'random_state': random_state,
            'n_jobs': -1
        }
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = []
        for train_idx, val_idx in skf.split(X_sub, y_sub):
            X_fold_train = X_sub.iloc[train_idx]
            y_fold_train = y_sub[train_idx]
            X_fold_val = X_sub.iloc[val_idx]
            y_fold_val = y_sub[val_idx]
            
            clf = xgb.XGBClassifier(**params)
            pipe = build_pipeline(clf, random_state=random_state)
            pipe.fit(X_fold_train, y_fold_train)
            
            val_probs = pipe.predict_proba(X_fold_val)[:, 1]
            val_preds = (val_probs >= 0.50).astype(int)
            scores.append(f1_score(y_fold_val, val_preds, zero_division=0))
        return np.mean(scores)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    
    print(f"XGBoost Best Trial: F1={study.best_value:.4f}")
    print(f"XGBoost Best Params: {study.best_params}")
    return study.best_params, study.best_value


def tune_lightgbm(X_train_df: pd.DataFrame, y_train: np.ndarray, n_trials: int = 10, cv_folds: int = 3, random_state: int = 42):
    """
    Bayesian optimization for LightGBM.
    Uses leakage-safe imbalanced-learn pipeline fitted independently per fold.
    """
    print(f"\n[Optuna] Tuning LightGBM ({n_trials} trials, {cv_folds}-fold CV)...")
    X_sub, y_sub = _subsample_if_large(X_train_df, y_train, random_state=random_state)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 80, 200, step=20),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'random_state': random_state,
            'verbose': -1,
            'n_jobs': -1
        }
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = []
        for train_idx, val_idx in skf.split(X_sub, y_sub):
            X_fold_train = X_sub.iloc[train_idx]
            y_fold_train = y_sub[train_idx]
            X_fold_val = X_sub.iloc[val_idx]
            y_fold_val = y_sub[val_idx]
            
            clf = lgb.LGBMClassifier(**params)
            pipe = build_pipeline(clf, random_state=random_state)
            pipe.fit(X_fold_train, y_fold_train)
            
            val_probs = pipe.predict_proba(X_fold_val)[:, 1]
            val_preds = (val_probs >= 0.50).astype(int)
            scores.append(f1_score(y_fold_val, val_preds, zero_division=0))
        return np.mean(scores)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    
    print(f"LightGBM Best Trial: F1={study.best_value:.4f}")
    print(f"LightGBM Best Params: {study.best_params}")
    return study.best_params, study.best_value
