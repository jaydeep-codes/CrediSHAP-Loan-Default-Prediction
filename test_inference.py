"""
test_inference.py - Inference Pipeline Test for CrediSHAP.

Tests the saved-model inference pipeline with multiple diverse applicant profiles
to verify that model artifacts load correctly and produce reasonable predictions.
Confirms that decision support notices and fairness notices are returned.
"""
import os
import sys

from src.predict import CrediSHAPPredictor

# Use correct relative models path
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
predictor = CrediSHAPPredictor(models_dir)

profiles = [
    {
        "name": "Applicant 1: Strong Profile",
        "data": {
            "loan_amount": 150000, "term": 360.0, "property_value": 450000,
            "income": 12000, "Credit_Score": 810, "LTV": 33.3, "dtir1": 22.0,
            "loan_limit": "cf", "Gender": "Male", "approv_in_adv": "pre",
            "loan_type": "type1", "loan_purpose": "p1", "Credit_Worthiness": "l1",
            "open_credit": "nopc", "business_or_commercial": "nob/c",
            "Neg_ammortization": "not_neg", "interest_only": "not_int",
            "lump_sum_payment": "not_lpsm", "construction_type": "sb",
            "occupancy_type": "pr", "Secured_by": "home", "total_units": "1U",
            "credit_type": "CIB", "co-applicant_credit_type": "CIB",
            "age": "35-44", "submission_of_application": "to_inst",
            "Region": "North", "Security_Type": "direct"
        }
    },
    {
        "name": "Applicant 2: High Risk Profile",
        "data": {
            "loan_amount": 500000, "term": 360.0, "property_value": 480000,
            "income": 2500, "Credit_Score": 540, "LTV": 104.0, "dtir1": 56.0,
            "loan_limit": "ncf", "Gender": "Joint", "approv_in_adv": "nopre",
            "loan_type": "type2", "loan_purpose": "p3", "Credit_Worthiness": "l2",
            "open_credit": "opc", "business_or_commercial": "nob/c",
            "Neg_ammortization": "neg_amm", "interest_only": "int_only",
            "lump_sum_payment": "not_lpsm", "construction_type": "sb",
            "occupancy_type": "ir", "Secured_by": "home", "total_units": "2U",
            "credit_type": "EQUI", "co-applicant_credit_type": "EXP",
            "age": "25-34", "submission_of_application": "not_inst",
            "Region": "south", "Security_Type": "direct"
        }
    }
]

print("--- Testing Applicant Inference ---")
all_passed = True
for p in profiles:
    try:
        res = predictor.predict_one(p["data"])
        prob = res['default_probability']
        print(f"\n{p['name']}:")
        print(f"  Probability: {prob*100:.2f}% | Category: {res['risk_category']}")
        print(f"  Decision Support Notice: {res['decision_support_notice']}")
        print(f"  Top Risk Factors: {[x['feature'] for x in res['risk_increasing_factors'][:2]]}")
        print(f"  Top Mitigating Factors: {[x['feature'] for x in res['risk_reducing_factors'][:2]]}")
        
        # Sanity assertions
        assert 0.0 <= prob <= 1.0, f"Probability {prob} out of range!"
        assert "decision_support_notice" in res, "Missing decision_support_notice!"
        assert "fairness_notice" in res, "Missing fairness_notice!"
    except Exception as e:
        print(f"\n{p['name']}: FAILED - {e}")
        all_passed = False

if all_passed:
    print("\nAll applicant inference tests passed successfully!")
else:
    print("\nSome inference tests FAILED. Review errors above.")
    sys.exit(1)
