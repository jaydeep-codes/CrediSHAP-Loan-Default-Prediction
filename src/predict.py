"""
predict.py - Production-Ready Prediction and Inference Pipeline for CrediSHAP.

Loads:
- models/model.pkl
- models/preprocessor.pkl
- models/threshold.json
- models/metadata.json

Provides real-time scoring, decision support categorization, and local SHAP explanations.
Clarifies that outputs are for decision support only, not automated lending decisions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


class CrediSHAPPredictor:
    """Predictor and Explainer service for CrediSHAP Decision Support."""
    
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        
        # Load artifacts
        self.model_path = os.path.join(models_dir, "model.pkl")
        self.prep_path = os.path.join(models_dir, "preprocessor.pkl")
        self.thresh_path = os.path.join(models_dir, "threshold.json")
        self.meta_path = os.path.join(models_dir, "metadata.json")
        
        if not os.path.exists(self.model_path) or not os.path.exists(self.prep_path):
            raise FileNotFoundError("Model or preprocessor artifacts not found.")
            
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.prep_path)
        
        with open(self.thresh_path, 'r', encoding='utf-8') as f:
            thresh_data = json.load(f)
            self.threshold = float(thresh_data.get("optimal_threshold", 0.38))
            
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        self.feature_names = self.preprocessor.feature_names
        self.clean_feature_names = [f.replace("[", "_").replace("]", "_").replace("<", "lt_").replace(">", "gt_") for f in self.feature_names]
        self.explainer = shap.TreeExplainer(self.model)

    def predict_one(self, applicant_data: dict):
        """
        Predict default probability and risk level for a single applicant.
        applicant_data: dict matching raw feature names.
        """
        df = pd.DataFrame([applicant_data])
        X_proc = self.preprocessor.transform(df)
        
        # Predict probability
        prob = float(self.model.predict_proba(X_proc)[0, 1])
        is_default = bool(prob >= self.threshold)
        
        # Decision support risk categorization (non-automated decision)
        if prob < self.threshold * 0.6:
            risk_category = "Lower predicted default risk"
            risk_color = "green"
        elif prob < self.threshold:
            risk_category = "Moderate risk (Manual review recommended)"
            risk_color = "orange"
        else:
            risk_category = "Higher predicted default risk"
            risk_color = "red"
            
        # Compute local SHAP attribution
        shap_res = self.explainer(X_proc)
        if len(shap_res.shape) == 3 and shap_res.shape[2] == 2:
            shap_vals = shap_res.values[0, :, 1]
            base_val = float(shap_res.base_values[0, 1])
        else:
            shap_vals = shap_res.values[0]
            base_val = float(shap_res.base_values[0]) if hasattr(shap_res.base_values, '__iter__') else float(shap_res.base_values)
            
        # Feature impact attribution
        impacts = []
        for name, val, sv in zip(self.clean_feature_names, X_proc[0], shap_vals):
            impacts.append({
                "feature": name,
                "value": float(val),
                "shap_impact": float(sv)
            })
            
        risk_increasing = sorted([x for x in impacts if x["shap_impact"] > 0], key=lambda x: x["shap_impact"], reverse=True)[:4]
        risk_reducing = sorted([x for x in impacts if x["shap_impact"] < 0], key=lambda x: x["shap_impact"])[:4]
        
        return {
            "default_probability": prob,
            "decision_threshold": self.threshold,
            "predicted_class": 1 if is_default else 0,
            "risk_category": risk_category,
            "risk_color": risk_color,
            "decision_support_notice": "Decision Support Notice: This assessment provides risk probability estimation and feature attributions for human underwriter assistance. It is NOT an automated lending decision or credit approval.",
            "fairness_notice": "Fairness Assessment Note: Contains Gender and Region features. Under Equal Credit Opportunity Act (ECOA) and fair lending regulations, protected demographic attributes should not serve as bases for adverse credit actions.",
            "base_value": base_val,
            "risk_increasing_factors": risk_increasing,
            "risk_reducing_factors": risk_reducing,
            "raw_shap_values": shap_vals.tolist(),
            "X_proc": X_proc
        }

    def generate_waterfall_plot(self, prediction_result: dict):
        """Generates matplotlib waterfall figure for the prediction."""
        shap_vals = np.array(prediction_result["raw_shap_values"])
        base_val = prediction_result["base_value"]
        X_proc = prediction_result["X_proc"]
        
        exp = shap.Explanation(
            values=shap_vals,
            base_values=base_val,
            data=X_proc[0],
            feature_names=self.clean_feature_names
        )
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(exp, max_display=8, show=False)
        plt.tight_layout()
        return fig
