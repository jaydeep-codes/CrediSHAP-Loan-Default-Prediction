# CrediSHAP: Explainable Machine Learning for Loan Default Prediction

**Subtitle:** A SHAP-Integrated, Class-Balance-Aware Machine Learning Framework for Real-Time, Interpretable Credit Risk Assessment with Model Selection

---

## 1. Project Overview

CrediSHAP is a research-grade, transparent credit risk classification framework developed to predict loan default probability while maintaining strict data leakage prevention, handling real-world class imbalance, optimizing classification thresholds, and providing both global and local game-theoretic SHAP explainability.

> **Decision Support Advisory:** CrediSHAP outputs default probability estimates and feature-level attributions to assist loan officers and credit underwriters. It is strictly an explainable decision-support tool, **NOT** an automated lending decision or credit approval engine.

---

## 2. Architecture & Model Selection Strategy

Rather than deploying a multi-model voting or stacking ensemble, CrediSHAP uses a disciplined **model-selection protocol**:
1. Evaluates multiple distinct tree-based architectures (**Random Forest**, **XGBoost**, and **LightGBM**) under identical, leakage-safe 3-fold Stratified Cross-Validation.
2. Selects a single **Champion Model** according to a pre-specified objective: **Highest CV F1-Score on the original, un-resampled training distribution**.
3. Optimizes the classification decision threshold strictly across out-of-fold (OOF) validation predictions without touching the test set.
4. Deploys the single fitted champion model with TreeExplainer SHAP attribution.

---

## 3. Repository Structure

```
CrediSHAP/
├── data/
│   └── dataset.csv             # Kaggle Loan Default Dataset (148,670 rows, 34 columns)
├── notebooks/
│   └── Loan.ipynb              # Baseline notebook
├── src/
│   ├── preprocessing.py        # Leakage-safe imblearn pipeline (imputation, OHE, SMOTE)
│   ├── tune.py                 # Bayesian optimization via Optuna (per-fold pipeline fitting)
│   ├── train.py                # 3-Fold CV model comparison, champion selection, OOF threshold tuning
│   ├── evaluate.py             # Isolated test evaluation, metrics computation, and plotting
│   ├── explain.py              # TreeExplainer SHAP summary, beeswarm, waterfall plots
│   └── predict.py              # Inference pipeline with advisory notice & fairness warnings
├── models/
│   ├── model.pkl               # Serialized champion model (XGBoost)
│   ├── preprocessor.pkl        # Serialized fitted CrediSHAPPreprocessor
│   ├── threshold.json          # Optimal locked classification threshold (0.35)
│   └── metadata.json           # Comprehensive machine-readable experimental metadata
├── outputs/
│   ├── confusion_matrix.png    # Test set confusion matrix
│   ├── roc_curve.png           # Multi-model ROC curves
│   ├── precision_recall_curve.png # Multi-model Precision-Recall curves
│   ├── model_comparison.png    # Comparative performance bar chart
│   ├── shap_summary.png        # Global mean |SHAP| feature importance
│   ├── shap_beeswarm.png       # Global directional beeswarm plot
│   └── local_explanation.png   # Local applicant waterfall explanation
├── app/
│   └── streamlit_app.py        # Interactive Streamlit decision-support dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation
└── main.py                     # Master end-to-end execution script
```

---

## 4. Experimental Results Summary

- **Total Records:** 148,670
- **Target Variable:** `Status` (0 = Non-default, 1 = Default)
- **Class Distribution:** Non-default: 112,031 (75.36%) | Default: 36,639 (24.64%)
- **Data Splitting:** Stratified 80:20 (118,936 training, 29,734 untouched test samples)
- **Oversampling Strategy:** SMOTE fitted strictly inside each training partition; validation partitions and final test set are never resampled.
- **Candidate Models Evaluated:** Random Forest, XGBoost, LightGBM
- **Champion Model Selected:** **XGBoost** (Highest 3-Fold Stratified CV F1-score: **0.7421**)
- **Optimal Decision Threshold:** **0.35** (Selected via OOF F1-maximization on the training set; locked pre-test)

### 3-Fold Cross-Validation Performance (Leakage-Safe Validation Distribution)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 87.95% | 94.38% | 54.43% | 0.6904 | 0.8827 | 0.8182 |
| **LightGBM** | 89.32% | 92.72% | 61.18% | 0.7372 | 0.8919 | 0.8363 |
| **XGBoost (Selected Champion)** | **89.36%** | **92.12%** | **62.14%** | **0.7421** | **0.8936** | **0.8382** |

### Final Test Set Performance (29,734 Untouched Samples)

| Model | Decision Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.50 | 88.58% | 95.24% | 56.48% | 0.7091 | 0.8878 | 0.8292 |
| **LightGBM** | 0.50 | 89.89% | 93.83% | 63.13% | 0.7548 | 0.8958 | 0.8447 |
| **XGBoost (Default)** | 0.50 | 89.86% | 93.09% | 63.56% | 0.7554 | 0.8979 | 0.8470 |
| **XGBoost (CrediSHAP Champion)** | **0.35** | **89.34%** | **85.38%** | **68.46%** | **0.7599** | **0.8979** | **0.8470** |

*Champion Test Set Confusion Matrix (Threshold 0.35):*
- **True Negatives (TN):** 21,548 | **False Positives (FP):** 858
- **False Negatives (FN):** 2,311 | **True Positives (TP):** 5,017
- Identifies **68.46%** of all defaulters while preserving an $85.38\%$ precision rate.

> **Note on Evaluation Validity:** High ROC-AUC (0.8979) measures strong discriminative ranking between defaulters and non-defaulters. The absence of data leakage and over-optimism is guaranteed by architectural pipeline isolation (fitting all imputers, encoders, and SMOTE exclusively on training partitions), not by the ROC-AUC score itself.

---

## 5. Fairness, Demographic Bias & Regulatory Limitations

In the public dataset, borrower `Gender` and geographic `Region` are present as features:
- `Region_south` emerges as a top-ranking attribution feature in the global SHAP analysis.
- Under the **Equal Credit Opportunity Act (ECOA, 15 U.S.C. § 1691 / CFPB Regulation B)**, creditors are legally prohibited from discriminating on the basis of sex/gender or marital status. Using gender in production underwriting models directly violates federal fair lending laws.
- Under the **Fair Housing Act (FHA, 42 U.S.C. § 3601 et seq.)**, geographic variables can act as unconstitutional proxies for racial or ethnic demographics, resulting in unlawful disparate impact (redlining).
- **Production Recommendation:** Prior to commercial deployment, sensitive demographic identifiers (`Gender`, `Region`) must be eliminated from candidate features, and models must be audited for Disparate Impact and Demographic Parity across protected groups.

---

## 6. How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Full Pipeline Execution
```bash
python main.py
```

### Test Saved-Model Inference
```bash
python test_inference.py
```

### Launch Interactive Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```
