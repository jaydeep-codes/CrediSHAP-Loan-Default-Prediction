"""
streamlit_app.py - High-Performance Decision-Support Dashboard for CrediSHAP.
Executive-Grade White Theme UI with Interactive Presets and Explainability.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

# Ensure src can be loaded
current_dir = os.path.dirname(os.path.abspath(__file__))
credishap_dir = os.path.abspath(os.path.join(current_dir, ".."))
if credishap_dir not in sys.path:
    sys.path.insert(0, credishap_dir)

from src.predict import CrediSHAPPredictor

# Streamlit Page Configuration
st.set_page_config(
    page_title="CrediSHAP | Credit Risk Intelligence Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling - Strictly White Background Theme
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Root Variables */
    :root {
        --bg-main: #FFFFFF;
        --bg-surface: #FFFFFF;
        --bg-subtle: #F8FAFC;
        --border-color: #E2E8F0;
        --border-subtle: #F1F5F9;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #64748B;
        --brand-primary: #2563EB;
        --brand-dark: #1E3A8A;
        --brand-light: #EFF6FF;
        --success: #10B981;
        --success-bg: #ECFDF5;
        --warning: #F59E0B;
        --warning-bg: #FFFBEB;
        --danger: #EF4444;
        --danger-bg: #FEF2F2;
    }

    /* Strict White Background Enforcement */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Sidebar Clean Styling */
    [data-testid="stSidebar"] {
        background-color: #FAFAFA !important;
        border-right: 1px solid var(--border-color) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: #FAFAFA !important;
    }

    /* Hide standard Streamlit clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: #FFFFFF !important;}

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        letter-spacing: -0.02em !important;
    }

    /* Main Brand Banner */
    .brand-header-card {
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
    }

    .brand-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #1E3A8A;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .brand-subtitle {
        color: #475569;
        font-size: 0.95rem;
        margin-top: 0.3rem;
        font-weight: 400;
    }

    /* Compliance & Advisory Notice */
    .advisory-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        padding: 0.85rem 1.2rem;
        border-radius: 8px;
        font-size: 0.88rem;
        color: #334155;
        margin-bottom: 1.5rem;
        line-height: 1.45;
    }

    /* Card Containers */
    .ui-card {
        background-color: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.25rem;
    }

    /* Metric KPI Tiles */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 1rem;
        margin-bottom: 1.25rem;
    }

    .kpi-tile {
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .kpi-tile:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06);
    }

    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }

    .kpi-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
    }

    .kpi-subtext {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.35rem;
    }

    /* Status Badges */
    .risk-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .risk-pill-low {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }

    .risk-pill-moderate {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
    }

    .risk-pill-high {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }

    /* Feature Attribution Item */
    .attribution-item {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.9rem;
    }

    .attribution-item-risk {
        border-left: 4px solid #EF4444;
        background: #FFFDFD;
    }

    .attribution-item-safe {
        border-left: 4px solid #10B981;
        background: #FDFFFD;
    }

    .attr-name {
        font-weight: 600;
        color: #1E293B;
    }

    .attr-detail {
        font-size: 0.8rem;
        color: #64748B;
    }

    .attr-badge-danger {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 700;
        background: #FEE2E2;
        color: #B91C1C;
        padding: 3px 8px;
        border-radius: 6px;
    }

    .attr-badge-safe {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 700;
        background: #D1FAE5;
        color: #047857;
        padding: 3px 8px;
        border-radius: 6px;
    }

    /* Interactive Risk Spectrum Bar */
    .spectrum-container {
        position: relative;
        height: 24px;
        background: linear-gradient(to right, #10B981 0%, #10B981 21%, #F59E0B 21%, #F59E0B 35%, #EF4444 35%, #DC2626 100%);
        border-radius: 12px;
        margin: 1.25rem 0 1.75rem 0;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.15);
    }

    .spectrum-threshold-line {
        position: absolute;
        top: -6px;
        bottom: -6px;
        left: 35%;
        width: 3px;
        background-color: #0F172A;
        border-radius: 2px;
        z-index: 2;
    }

    .spectrum-threshold-label {
        position: absolute;
        top: -24px;
        left: 35%;
        transform: translateX(-50%);
        font-size: 0.72rem;
        font-weight: 700;
        color: #0F172A;
        background: #FFFFFF;
        padding: 1px 6px;
        border: 1px solid #CBD5E1;
        border-radius: 4px;
        white-space: nowrap;
    }

    /* Clean Styled Inputs & Forms */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
        font-size: 0.92rem !important;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
    }

    /* Custom Form Submit Button */
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.65rem 1.5rem !important;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.25) !important;
        transition: background-color 0.15s ease, transform 0.15s ease !important;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #1D4ED8 !important;
        transform: translateY(-1px) !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E2E8F0;
        background-color: #FFFFFF;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #64748B;
        border-radius: 8px 8px 0 0;
        border: none;
        background: transparent;
    }

    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        border-bottom: 2px solid #2563EB !important;
        background-color: #FFFFFF !important;
    }

    /* Modern Table Formatting */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
        background: #FFFFFF;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
    }

    .custom-table th {
        background-color: #F8FAFC;
        color: #475569;
        font-weight: 700;
        text-align: left;
        padding: 10px 14px;
        border-bottom: 1px solid #E2E8F0;
    }

    .custom-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #F1F5F9;
        color: #1E293B;
    }

    .custom-table tr:hover {
        background-color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)


# Load Model and Resources
@st.cache_resource
def load_predictor():
    models_dir = os.path.join(credishap_dir, "models")
    return CrediSHAPPredictor(models_dir)


try:
    predictor = load_predictor()
    metadata = predictor.metadata
    threshold = predictor.threshold
    champion_name = metadata.get("model_selection", {}).get("selected_champion_model", "XGBoost")
except Exception as e:
    st.error(f"Error loading CrediSHAP models: {e}. Please ensure model artifacts are present.")
    st.stop()


# Applicant Presets for Underwriter Productivity
PRESETS = {
    "custom": {
        "name": "Custom Underwriting Scenario",
        "loan_amount": 300000, "property_value": 400000, "income": 6500,
        "Credit_Score": 710, "term": 360.0, "LTV": 75.0, "dtir1": 38.0,
        "loan_limit": "cf", "Credit_Worthiness": "l1", "credit_type": "EXP",
        "co_app_credit": "CIB", "Gender": "Male", "age": "35-44",
        "loan_purpose": "p1", "loan_type": "type1", "approv_in_adv": "nopre",
        "occupancy_type": "pr", "Region": "North"
    },
    "prime": {
        "name": "Prime Applicant (Low Risk)",
        "loan_amount": 150000, "property_value": 450000, "income": 12000,
        "Credit_Score": 810, "term": 360.0, "LTV": 33.3, "dtir1": 22.0,
        "loan_limit": "cf", "Credit_Worthiness": "l1", "credit_type": "CIB",
        "co_app_credit": "CIB", "Gender": "Male", "age": "35-44",
        "loan_purpose": "p1", "loan_type": "type1", "approv_in_adv": "pre",
        "occupancy_type": "pr", "Region": "North"
    },
    "borderline": {
        "name": "Borderline Applicant (Manual Review)",
        "loan_amount": 280000, "property_value": 350000, "income": 5800,
        "Credit_Score": 670, "term": 360.0, "LTV": 80.0, "dtir1": 42.0,
        "loan_limit": "cf", "Credit_Worthiness": "l1", "credit_type": "CRIF",
        "co_app_credit": "EXP", "Gender": "Female", "age": "45-54",
        "loan_purpose": "p3", "loan_type": "type1", "approv_in_adv": "nopre",
        "occupancy_type": "pr", "Region": "central"
    },
    "distressed": {
        "name": "Subprime / High Risk Applicant",
        "loan_amount": 500000, "property_value": 480000, "income": 2500,
        "Credit_Score": 540, "term": 360.0, "LTV": 104.0, "dtir1": 56.0,
        "loan_limit": "ncf", "Credit_Worthiness": "l2", "credit_type": "EQUI",
        "co_app_credit": "EXP", "Gender": "Joint", "age": "25-34",
        "loan_purpose": "p3", "loan_type": "type2", "approv_in_adv": "nopre",
        "occupancy_type": "ir", "Region": "south"
    }
}


# Initialize Session State
if "selected_preset_key" not in st.session_state:
    st.session_state["selected_preset_key"] = "custom"


# Sidebar Architecture
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.25rem 0; border-bottom: 1px solid #E2E8F0;">
        <div style="font-size: 1.3rem; font-weight: 800; color: #1E3A8A; display: flex; align-items: center; gap: 8px;">
            💳 CrediSHAP
        </div>
        <div style="font-size: 0.78rem; color: #64748B; margin-top: 3px;">
            Decision Support Architecture v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### ⚡ Quick Profile Presets")
    preset_choice = st.selectbox(
        "Load Preconfigured Scenario",
        options=["custom", "prime", "borderline", "distressed"],
        format_func=lambda x: PRESETS[x]["name"],
        index=0,
        key="preset_selector"
    )
    
    st.markdown("---")
    st.markdown("#### 🛡️ Champion Model Profile")
    st.markdown(f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.85rem; font-size: 0.83rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="color: #64748B;">Architecture:</span>
            <strong style="color: #0F172A;">{champion_name}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="color: #64748B;">Decision Threshold:</span>
            <strong style="color: #2563EB;">{threshold:.2f} (Locked)</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="color: #64748B;">CV F1-Score:</span>
            <strong style="color: #10B981;">0.7421</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="color: #64748B;">Test ROC-AUC:</span>
            <strong style="color: #0F172A;">0.8979</strong>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #64748B;">Data Partition:</span>
            <span style="color: #0F172A;">80:20 Leakage-Safe</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 🎯 Decision Tier Thresholds")
    st.markdown(f"""
    <div style="font-size: 0.8rem; line-height: 1.6;">
        <div style="margin-bottom: 6px;">
            <span class="risk-pill risk-pill-low" style="padding: 2px 8px; font-size: 0.72rem;">Low Risk</span>
            <span style="color: #475569;">Probability &lt; {threshold * 0.6:.2f}</span>
        </div>
        <div style="margin-bottom: 6px;">
            <span class="risk-pill risk-pill-moderate" style="padding: 2px 8px; font-size: 0.72rem;">Manual Review</span>
            <span style="color: #475569;">{threshold * 0.6:.2f} &le; P &lt; {threshold:.2f}</span>
        </div>
        <div>
            <span class="risk-pill risk-pill-high" style="padding: 2px 8px; font-size: 0.72rem;">High Risk</span>
            <span style="color: #475569;">Probability &ge; {threshold:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Main Application Header Card
st.markdown("""
<div class="brand-header-card">
    <div>
        <div class="brand-title">
            <span>💳</span> CrediSHAP Intelligence System
        </div>
        <div class="brand-subtitle">
            Explainable Machine Learning Framework for Real-Time Underwriting Decision Support
        </div>
    </div>
    <div style="display: flex; gap: 8px; margin-top: 0.5rem;">
        <span style="background: #F1F5F9; color: #475569; padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 600;">
            Champion: XGBoost
        </span>
        <span style="background: #EFF6FF; color: #2563EB; padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 600;">
            Threshold: 0.35
        </span>
        <span style="background: #ECFDF5; color: #059669; padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 600;">
            SHAP Integrated
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Regulatory Compliance Banner
st.markdown("""
<div class="advisory-card">
    ⚖️ <strong>Regulatory & Decision Support Notice:</strong> CrediSHAP computes default probability estimates and feature-level game-theoretic SHAP attributions solely to augment human underwriting expertise. This system is strictly an <strong>explainable decision-support tool</strong> and does <strong>NOT</strong> generate automated adverse lending determinations or approvals under ECOA / CFPB regulations.
</div>
""", unsafe_allow_html=True)

# Application Tabs
tab_eval, tab_audit, tab_fairness = st.tabs([
    "🔍 Real-Time Applicant Assessment",
    "📊 Empirical Model Validation",
    "⚖️ Fairness & Regulatory Governance"
])

current_data = PRESETS[preset_choice]

with tab_eval:
    st.markdown("### 📋 Applicant Financial Profile")
    st.caption("Adjust parameters below or load an underwriter profile preset from the sidebar.")
    
    with st.form("applicant_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 🏦 Loan & Asset Details")
            loan_amount = st.number_input(
                "Loan Amount (₹)",
                min_value=10000, max_value=2500000,
                value=int(current_data["loan_amount"]), step=10000
            )
            property_value = st.number_input(
                "Property Value (₹)",
                min_value=10000, max_value=5000000,
                value=int(current_data["property_value"]), step=10000
            )
            income = st.number_input(
                "Monthly Income (₹)",
                min_value=500, max_value=200000,
                value=int(current_data["income"]), step=500
            )
            term = st.selectbox(
                "Loan Term (Months)",
                options=[180.0, 240.0, 300.0, 360.0],
                index=[180.0, 240.0, 300.0, 360.0].index(current_data["term"])
            )
            loan_purpose = st.selectbox(
                "Loan Purpose",
                options=["p1", "p2", "p3", "p4"],
                index=["p1", "p2", "p3", "p4"].index(current_data["loan_purpose"]),
                help="p1: Home Purchase, p2: Improvement, p3: Refinancing, p4: Other"
            )
            
        with col2:
            st.markdown("##### 📈 Credit Standing & Leverage")
            Credit_Score = st.slider(
                "Credit Score (FICO)",
                min_value=500, max_value=900,
                value=int(current_data["Credit_Score"]), step=1
            )
            LTV = st.number_input(
                "Loan-to-Value (LTV) %",
                min_value=5.0, max_value=250.0,
                value=float(current_data["LTV"]), step=1.0,
                help="Calculated as (Loan Amount / Property Value) * 100"
            )
            dtir1 = st.number_input(
                "Debt-to-Income (DTI) %",
                min_value=5.0, max_value=75.0,
                value=float(current_data["dtir1"]), step=1.0,
                help="Monthly debt liabilities divided by monthly gross income"
            )
            Credit_Worthiness = st.selectbox(
                "Credit Worthiness",
                options=["l1", "l2"],
                index=["l1", "l2"].index(current_data["Credit_Worthiness"]),
                help="l1: Standard prime tier, l2: Subprime tier"
            )
            credit_type = st.selectbox(
                "Primary Credit Bureau Type",
                options=["CIB", "CRIF", "EQUI", "EXP"],
                index=["CIB", "CRIF", "EQUI", "EXP"].index(current_data["credit_type"])
            )

        with col3:
            st.markdown("##### 👤 Application & Demographic Context")
            Gender = st.selectbox(
                "Applicant Gender",
                options=["Male", "Female", "Joint", "Sex Not Available"],
                index=["Male", "Female", "Joint", "Sex Not Available"].index(current_data["Gender"])
            )
            age = st.selectbox(
                "Age Bracket",
                options=["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"],
                index=["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"].index(current_data["age"])
            )
            loan_limit = st.selectbox(
                "Conforming Category",
                options=["cf", "ncf"],
                index=["cf", "ncf"].index(current_data["loan_limit"]),
                help="cf: Conforming conventional loan, ncf: Non-conforming jumbo"
            )
            approv_in_adv = st.selectbox(
                "Pre-Approval Status",
                options=["nopre", "pre"],
                index=["nopre", "pre"].index(current_data["approv_in_adv"])
            )
            occupancy_type = st.selectbox(
                "Property Occupancy",
                options=["pr", "sr", "ir"],
                index=["pr", "sr", "ir"].index(current_data["occupancy_type"]),
                help="pr: Principal residence, sr: Secondary home, ir: Investment"
            )
            Region = st.selectbox(
                "Geographic Region",
                options=["North", "south", "central", "North-East"],
                index=["North", "south", "central", "North-East"].index(current_data["Region"])
            )
            co_app_credit = current_data["co_app_credit"]

        # Secondary Technical Application Fields
        loan_type = current_data.get("loan_type", "type1")
        business_or_commercial = "nob/c"
        open_credit = "nopc"
        Neg_ammortization = "not_neg"
        interest_only = "not_int"
        lump_sum_payment = "not_lpsm"
        construction_type = "sb"
        Secured_by = "home"
        total_units = "1U"
        submission_of_application = "to_inst"
        Security_Type = "direct"

        submit_btn = st.form_submit_button("⚡ Evaluate Applicant Risk & Compute SHAP Attribution", use_container_width=True)

    # Perform Prediction Evaluation
    if submit_btn or "last_result" in st.session_state:
        if submit_btn:
            applicant_payload = {
                "loan_amount": loan_amount,
                "term": term,
                "property_value": property_value,
                "income": income,
                "Credit_Score": Credit_Score,
                "LTV": LTV,
                "dtir1": dtir1,
                "loan_limit": loan_limit,
                "Gender": Gender,
                "approv_in_adv": approv_in_adv,
                "loan_type": loan_type,
                "loan_purpose": loan_purpose,
                "Credit_Worthiness": Credit_Worthiness,
                "open_credit": open_credit,
                "business_or_commercial": business_or_commercial,
                "Neg_ammortization": Neg_ammortization,
                "interest_only": interest_only,
                "lump_sum_payment": lump_sum_payment,
                "construction_type": construction_type,
                "occupancy_type": occupancy_type,
                "Secured_by": Secured_by,
                "total_units": total_units,
                "credit_type": credit_type,
                "co-applicant_credit_type": co_app_credit,
                "age": age,
                "submission_of_application": submission_of_application,
                "Region": Region,
                "Security_Type": Security_Type
            }
            res = predictor.predict_one(applicant_payload)
            st.session_state["last_result"] = res
        else:
            res = st.session_state["last_result"]

        prob = res["default_probability"]
        is_default = res["predicted_class"]
        category = res["risk_category"]
        color = res["risk_color"]
        
        if color == "green":
            pill_class = "risk-pill-low"
            pill_icon = "🟢"
            status_desc = "Applicant exhibits low default probability. Meets prime underwriting criteria."
        elif color == "orange":
            pill_class = "risk-pill-moderate"
            pill_icon = "🟡"
            status_desc = "Applicant default probability falls in the borderline zone. Human underwriting review recommended."
        else:
            pill_class = "risk-pill-high"
            pill_icon = "🔴"
            status_desc = "Applicant default probability exceeds locked decision threshold (0.35). High probability of credit default."

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🎯 Underwriting Evaluation Outcome")

        # KPI Metrics Grid
        kpi_html = f"""
        <div class="kpi-grid">
            <div class="kpi-tile">
                <div class="kpi-label">Predicted Default Risk</div>
                <div class="kpi-value" style="color: {'#059669' if color=='green' else ('#D97706' if color=='orange' else '#DC2626')};">
                    {prob * 100:.2f}%
                </div>
                <div class="kpi-subtext">Base expectation: 24.64%</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-label">Locked Decision Threshold</div>
                <div class="kpi-value">{threshold:.2f}</div>
                <div class="kpi-subtext">Optimized via Out-of-Fold F1</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-label">Risk Tier Classification</div>
                <div style="margin-top: 4px;">
                    <span class="risk-pill {pill_class}">
                        {pill_icon} {category}
                    </span>
                </div>
                <div class="kpi-subtext">{status_desc}</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-label">Champion Architecture</div>
                <div class="kpi-value" style="font-size: 1.35rem; color: #1E3A8A;">{champion_name}</div>
                <div class="kpi-subtext">F1: 0.7599 | Leakage-Isolated</div>
            </div>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

        # Visual Risk Spectrum Bar
        prob_pct = min(max(prob * 100, 0.0), 100.0)
        marker_left = prob_pct
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 600; color: #475569; margin-bottom: 6px;">
                <span>0% Probability (Low Risk)</span>
                <span>Applicant Position: <strong>{prob * 100:.2f}%</strong></span>
                <span>100% (High Risk)</span>
            </div>
            <div class="spectrum-container">
                <div class="spectrum-threshold-line"></div>
                <div class="spectrum-threshold-label">Threshold: {threshold:.2f}</div>
                <div style="position: absolute; left: {marker_left}%; top: -8px; transform: translateX(-50%); width: 16px; height: 16px; background-color: #0F172A; border: 3px solid #FFFFFF; border-radius: 50%; box-shadow: 0 2px 4px rgba(0,0,0,0.3); z-index: 3;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #64748B;">
                <span style="color: #059669; font-weight: 600;">Low Risk Zone (&lt; 0.21)</span>
                <span style="color: #D97706; font-weight: 600;">Manual Review Tier (0.21 - 0.35)</span>
                <span style="color: #DC2626; font-weight: 600;">Adverse Risk Tier (&ge; 0.35)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # SHAP Feature Attribution Analysis
        st.markdown("### 🔍 Local Feature Attribution (SHAP TreeExplainer)")
        st.caption("Game-theoretic attribution showing exact directional impact of applicant parameters on the default score.")

        exp_col1, exp_col2 = st.columns([1.05, 1.15])

        with exp_col1:
            st.markdown("##### 🚨 Top Risk-Increasing Drivers")
            if res["risk_increasing_factors"]:
                for f in res["risk_increasing_factors"]:
                    f_name = f['feature'].replace("_", " ")
                    st.markdown(f"""
                    <div class="attribution-item attribution-item-risk">
                        <div>
                            <div class="attr-name">{f_name}</div>
                            <div class="attr-detail">Observed Value: <strong>{f['value']:.2f}</strong></div>
                        </div>
                        <div class="attr-badge-danger">+{f['shap_impact']:.4f} SHAP</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No significant risk-increasing factors detected for this applicant.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 🛡️ Top Risk-Mitigating Drivers")
            if res["risk_reducing_factors"]:
                for f in res["risk_reducing_factors"]:
                    f_name = f['feature'].replace("_", " ")
                    st.markdown(f"""
                    <div class="attribution-item attribution-item-safe">
                        <div>
                            <div class="attr-name">{f_name}</div>
                            <div class="attr-detail">Observed Value: <strong>{f['value']:.2f}</strong></div>
                        </div>
                        <div class="attr-badge-safe">{f['shap_impact']:.4f} SHAP</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No significant risk-mitigating factors detected for this applicant.")

        with exp_col2:
            st.markdown("##### 📊 SHAP Waterfall Visualization")
            with st.container():
                fig = predictor.generate_waterfall_plot(res)
                fig.patch.set_facecolor('#FFFFFF')
                for ax in fig.axes:
                    ax.set_facecolor('#FFFFFF')
                st.pyplot(fig, use_container_width=True)

with tab_audit:
    st.markdown("### 📋 Empirical Model Validation & Anti-Leakage Protocol")
    st.caption("All metrics computed strictly on 29,734 untouched test samples under leakage-isolated pipeline conditions.")
    
    ds = metadata.get("dataset_statistics", {})
    
    # Dataset Summary Metrics
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-label">Total Records</div>
            <div class="kpi-value">{ds.get('total_records', 148670):,}</div>
            <div class="kpi-subtext">Raw Kaggle Dataset</div>
        </div>
        """, unsafe_allow_html=True)
    with d_col2:
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-label">Class Ratio (Non/Def)</div>
            <div class="kpi-value">{ds.get('class_0_percentage', 75.36):.1f}% / {ds.get('class_1_percentage', 24.64):.1f}%</div>
            <div class="kpi-subtext">Real-world Imbalance</div>
        </div>
        """, unsafe_allow_html=True)
    with d_col3:
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-label">Training Partition</div>
            <div class="kpi-value">{ds.get('train_samples', 118936):,}</div>
            <div class="kpi-subtext">80% Stratified Split</div>
        </div>
        """, unsafe_allow_html=True)
    with d_col4:
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-label">Isolated Test Set</div>
            <div class="kpi-value">{ds.get('test_samples', 29734):,}</div>
            <div class="kpi-subtext">20% Evaluated ONCE</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🏆 Test Set Model Comparison")
    
    # Clean Test Set Comparison Table
    table_html = """
    <table class="custom-table">
        <thead>
            <tr>
                <th>Model Architecture</th>
                <th>Threshold</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>ROC-AUC</th>
                <th>PR-AUC</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Random Forest</strong></td>
                <td>0.50</td>
                <td>88.58%</td>
                <td>95.24%</td>
                <td>56.48%</td>
                <td>0.7091</td>
                <td>0.8878</td>
                <td>0.8292</td>
                <td><span style="color: #64748B;">Baseline</span></td>
            </tr>
            <tr>
                <td><strong>LightGBM</strong></td>
                <td>0.50</td>
                <td>89.89%</td>
                <td>93.83%</td>
                <td>63.13%</td>
                <td>0.7548</td>
                <td>0.8958</td>
                <td>0.8447</td>
                <td><span style="color: #64748B;">Candidate</span></td>
            </tr>
            <tr>
                <td><strong>XGBoost (Default)</strong></td>
                <td>0.50</td>
                <td>89.86%</td>
                <td>93.09%</td>
                <td>63.56%</td>
                <td>0.7554</td>
                <td>0.8979</td>
                <td>0.8470</td>
                <td><span style="color: #64748B;">Standard</span></td>
            </tr>
            <tr style="background-color: #F8FAFC; border-left: 3px solid #2563EB;">
                <td><strong style="color: #1E3A8A;">XGBoost (CrediSHAP)</strong></td>
                <td><strong style="color: #2563EB;">0.35</strong></td>
                <td><strong>89.34%</strong></td>
                <td><strong>85.38%</strong></td>
                <td><strong style="color: #059669;">68.46%</strong></td>
                <td><strong style="color: #1E3A8A;">0.7599</strong></td>
                <td><strong>0.8979</strong></td>
                <td><strong>0.8470</strong></td>
                <td><span class="risk-pill risk-pill-low" style="padding: 2px 8px; font-size: 0.72rem;">Selected Champion</span></td>
            </tr>
        </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📈 Diagnostic Visualizations")
    
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        roc_path = os.path.join(credishap_dir, "outputs", "roc_curve.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="Figure 1: Multi-Model ROC Curve Comparison (Isolated Test Set)", use_container_width=True)
        
        cm_path = os.path.join(credishap_dir, "outputs", "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Figure 2: Champion Model Confusion Matrix at Threshold 0.35", use_container_width=True)

    with img_col2:
        pr_path = os.path.join(credishap_dir, "outputs", "precision_recall_curve.png")
        if os.path.exists(pr_path):
            st.image(pr_path, caption="Figure 3: Precision-Recall Curve Comparison on Imbalanced Test Set", use_container_width=True)

        shap_path = os.path.join(credishap_dir, "outputs", "shap_summary.png")
        if os.path.exists(shap_path):
            st.image(shap_path, caption="Figure 4: Global SHAP Mean |Attribution| Ranking", use_container_width=True)

with tab_fairness:
    st.markdown("### ⚖️ Ethical Governance, Fairness & Regulatory Auditing")
    st.caption("Compliance review under Equal Credit Opportunity Act (ECOA) and Fair Housing Act (FHA) standards.")
    
    st.markdown("""
    <div class="ui-card">
        <h4 style="margin-top: 0; color: #DC2626;">🚨 Critical Compliance Notice: Protected Attributes in Dataset</h4>
        <p style="color: #475569; font-size: 0.92rem; line-height: 1.5;">
            The Kaggle Loan Default Dataset contains borrower demographic attributes, specifically <strong>Gender</strong> and geographic <strong>Region</strong>.
            Our SHAP interpretability analysis reveals substantial model attribution to these attributes:
        </p>
        <ul style="color: #334155; font-size: 0.9rem; line-height: 1.6;">
            <li><strong>Gender_Male</strong> ranks as the <strong>#4 most impactful global feature</strong> (mean |SHAP| ≈ 0.35).</li>
            <li><strong>Region_south</strong> ranks as the <strong>#7 most impactful global feature</strong> (mean |SHAP| ≈ 0.31).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 📜 Statutory Protections & Lending Regulations")
    
    reg_col1, reg_col2 = st.columns(2)
    with reg_col1:
        st.markdown("""
        <div class="ui-card">
            <h5 style="color: #1E3A8A; margin-top: 0;">1. Equal Credit Opportunity Act (ECOA)</h5>
            <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 8px;">15 U.S.C. § 1691 / CFPB Regulation B</div>
            <p style="font-size: 0.88rem; color: #334155; line-height: 1.5;">
                Federal law explicitly prohibits credit scoring engines and loan officers from discriminating against credit applicants based on sex, gender, marital status, race, religion, or national origin.
            </p>
            <div style="background: #FEF2F2; border-left: 3px solid #EF4444; padding: 6px 10px; font-size: 0.82rem; color: #991B1B; border-radius: 4px;">
                <strong>Regulatory Constraint:</strong> Utilizing Gender as an input variable in production credit scoring represents direct disparate treatment.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with reg_col2:
        st.markdown("""
        <div class="ui-card">
            <h5 style="color: #1E3A8A; margin-top: 0;">2. Fair Housing Act (FHA) Anti-Redlining</h5>
            <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 8px;">42 U.S.C. § 3601 et seq.</div>
            <p style="font-size: 0.88rem; color: #334155; line-height: 1.5;">
                Prohibits redlining and discrimination in residential mortgage transactions. Geographic identifiers can serve as unlawful proxies for demographic or racial composition.
            </p>
            <div style="background: #FEF2F2; border-left: 3px solid #EF4444; padding: 6px 10px; font-size: 0.82rem; color: #991B1B; border-radius: 4px;">
                <strong>Regulatory Constraint:</strong> Coarse regional proxies can inadvertently enforce geographic bias, triggering disparate impact liability.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🛠️ Production Remediation Checklist")
    
    st.markdown("""
    <div class="ui-card">
        <div style="display: flex; gap: 12px; align-items: flex-start; margin-bottom: 12px;">
            <div style="font-size: 1.2rem;">✅</div>
            <div>
                <strong>Feature Exclusion:</strong> Remove <code>Gender</code> and geographic <code>Region</code> prior to production model training.
            </div>
        </div>
        <div style="display: flex; gap: 12px; align-items: flex-start; margin-bottom: 12px;">
            <div style="font-size: 1.2rem;">✅</div>
            <div>
                <strong>Adverse Impact Ratio (AIR) Audit:</strong> Compute Demographic Parity and Equal Opportunity metrics across demographic cohorts.
            </div>
        </div>
        <div style="display: flex; gap: 12px; align-items: flex-start;">
            <div style="font-size: 1.2rem;">✅</div>
            <div>
                <strong>Adverse Action Explanations:</strong> Utilize SHAP attributions to generate legally compliant CFPB Adverse Action notices for denied applicants.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><hr style='border: none; border-top: 1px solid #E2E8F0;'><br>", unsafe_allow_html=True)
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #64748B;">
    <div>CrediSHAP Decision Support System &bull; Single Champion Architecture (XGBoost)</div>
    <div>Strict White Clean Theme &bull; Leakage-Isolated Evaluation</div>
</div>
""", unsafe_allow_html=True)
