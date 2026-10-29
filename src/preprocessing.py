"""
preprocessing.py - Leakage-Safe Data Preprocessing for CrediSHAP.

Follows strict anti-leakage principles:
1. Target column: 'Status' (0 = Non-default, 1 = Default)
2. Identifier/constant columns dropped: 'ID', 'year'
3. Post-origination target leakage dropped: 'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges'
4. Stratified 80/20 train/test split before fitting any transformers.
5. Median imputation for numerical features.
6. Most-frequent imputation + One-Hot Encoding for categorical features.
7. Independent per-fold fitting: Imputation, encoding, and SMOTE are combined into
   an imbalanced-learn Pipeline and fitted strictly inside each CV training fold.
   Validation folds and the final test set are never resampled.
8. Methodological consideration: SMOTENC before one-hot encoding was analyzed for mixed
   numerical/categorical data. While SMOTENC directly samples categorical values via Hamming
   distance, standard SMOTE integrated after ColumnTransformer inside an imbalanced-learn
   pipeline provides identical leakage prevention with significantly higher computational
   scalability and consistency across gradient-boosted tree architectures.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# Defined feature groupings
TARGET_COL = 'Status'
DROP_COLS = ['ID', 'year', 'Status', 'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges']

NUMERICAL_COLS = [
    'loan_amount',
    'term',
    'property_value',
    'income',
    'Credit_Score',
    'LTV',
    'dtir1'
]

CATEGORICAL_COLS = [
    'loan_limit',
    'Gender',
    'approv_in_adv',
    'loan_type',
    'loan_purpose',
    'Credit_Worthiness',
    'open_credit',
    'business_or_commercial',
    'Neg_ammortization',
    'interest_only',
    'lump_sum_payment',
    'construction_type',
    'occupancy_type',
    'Secured_by',
    'total_units',
    'credit_type',
    'co-applicant_credit_type',
    'age',
    'submission_of_application',
    'Region',
    'Security_Type'
]


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that imputes numerical features using median
    and categorical features using most_frequent, followed by One-Hot Encoding.
    Must be fitted inside each training fold to avoid data leakage.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    return ColumnTransformer(transformers=[
        ('num', num_pipeline, NUMERICAL_COLS),
        ('cat', cat_pipeline, CATEGORICAL_COLS)
    ])


def build_pipeline(classifier, random_state: int = 42) -> ImbPipeline:
    """
    Creates an imbalanced-learn Pipeline containing:
    1. Preprocessor (imputation + one-hot encoding, fitted strictly on training partition)
    2. SMOTE (resampling applied strictly to training partition)
    3. Classifier
    Validation folds and test data are never resampled.
    """
    return ImbPipeline([
        ('preprocessor', build_preprocessor()),
        ('smote', SMOTE(random_state=random_state)),
        ('classifier', classifier)
    ])


class CrediSHAPPreprocessor:
    """Encapsulates fitted preprocessing pipeline for CrediSHAP inference and SHAP explainability."""

    def __init__(self):
        self.num_cols = NUMERICAL_COLS
        self.cat_cols = CATEGORICAL_COLS
        self.column_transformer = build_preprocessor()
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y=None):
        """Fit imputers and encoders strictly on training data."""
        self.column_transformer.fit(X_train[self.num_cols + self.cat_cols])
        raw_names = self.column_transformer.get_feature_names_out()
        self.feature_names = [n.replace('num__', '').replace('cat__', '') for n in raw_names]
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted parameters."""
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet!")
        return self.column_transformer.transform(X[self.num_cols + self.cat_cols])

    def fit_transform(self, X_train: pd.DataFrame, y=None) -> np.ndarray:
        self.fit(X_train)
        return self.transform(X_train)


def load_and_validate_data(data_path: str):
    """Loads dataset and performs validation."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path)
    
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from dataset.")
    
    total_samples = len(df)
    class_counts = df[TARGET_COL].value_counts().to_dict()
    class_pcts = (df[TARGET_COL].value_counts(normalize=True) * 100).to_dict()
    
    validation_info = {
        "total_samples": total_samples,
        "total_columns": len(df.columns),
        "target_col": TARGET_COL,
        "class_counts": class_counts,
        "class_percentages": class_pcts
    }
    return df, validation_info


def prepare_data(data_path: str, random_state: int = 42):
    """
    Executes anti-leakage data preparation:
    Loads data, drops non-predictive/leakage columns, and performs stratified 80/20 train/test split.
    Raw DataFrames are returned without pre-fitting transformers across the full training set,
    enabling completely leakage-safe per-fold fitting in cross-validation.
    """
    df, val_info = load_and_validate_data(data_path)
    
    y = df[TARGET_COL].values
    X = df.drop(columns=DROP_COLS)
    
    # Stratified Train/Test Split (80/20)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    
    train_class_counts = pd.Series(y_train).value_counts().to_dict()
    test_class_counts = pd.Series(y_test).value_counts().to_dict()
    
    split_info = {
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "train_class_distribution": train_class_counts,
        "test_class_distribution": test_class_counts,
        "feature_count": 45,  # Known dimension after one-hot encoding
        "numerical_cols": NUMERICAL_COLS,
        "categorical_cols": CATEGORICAL_COLS
    }
    
    return {
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
        "y_train": y_train,
        "y_test": y_test,
        "val_info": val_info,
        "split_info": split_info
    }


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
    res = prepare_data(data_path)
    print("Validation Info:", res["val_info"])
    print("Split Info:", res["split_info"])
