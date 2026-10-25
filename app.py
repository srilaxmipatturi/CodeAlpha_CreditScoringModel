"""
app.py
------
Interactive Streamlit Web Dashboard for Credit Scoring and Loan Risk Assessment.
CodeAlpha Internship Project.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Set MPLCONFIGDIR
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())

# Workspace setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_engineering import engineer_financial_features, get_column_definitions
from src.scorecard import CreditScorecard

# Page Configuration
st.set_page_config(
    page_title="Credit Scoring & Risk Model | CodeAlpha",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.06);
        border-left: 5px solid #1e88e5;
    }
    .score-badge {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }
    .tier-badge {
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
    all_models = joblib.load(os.path.join(models_dir, 'all_models.pkl'))
    
    # Ensure cross-version scikit-learn compatibility
    for m in all_models.values():
        if hasattr(m, '__class__') and m.__class__.__name__ == 'LogisticRegression':
            if not hasattr(m, 'multi_class'):
                m.multi_class = 'auto'
    
    with open(os.path.join(models_dir, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)
        
    return preprocessor, all_models, metadata


try:
    preprocessor, all_models, metadata = load_all_artifacts()
    champion_name = metadata.get("champion_model", "Random Forest")
except Exception as e:
    st.error(f"Error loading models. Make sure you ran 'python main.py' first!\nDetails: {e}")
    st.stop()

scorecard = CreditScorecard()

# Preset Applicants for Quick Testing
PRESETS = {
    "Select Preset...": None,
    "🌟 Prime Borrower (Low Risk)": {
        "person_age": 38,
        "person_income": 95000,
        "person_home_ownership": "MORTGAGE",
        "person_emp_length": 8,
        "loan_intent": "HOMEIMPROVEMENT",
        "loan_amnt": 12000,
        "loan_int_rate": 8.5,
        "cb_person_cred_hist_length": 14,
        "total_credit_limit": 60000,
        "total_credit_used": 7500,
        "existing_loans_count": 2,
        "previous_loan_defaults_on_file": "N",
        "cb_person_delinquencies": 0,
        "cb_person_bankruptcies": 0
    },
    "⚖️ Average Borrower (Moderate Risk)": {
        "person_age": 30,
        "person_income": 52000,
        "person_home_ownership": "RENT",
        "person_emp_length": 4,
        "loan_intent": "PERSONAL",
        "loan_amnt": 14000,
        "loan_int_rate": 14.2,
        "cb_person_cred_hist_length": 6,
        "total_credit_limit": 25000,
        "total_credit_used": 12000,
        "existing_loans_count": 3,
        "previous_loan_defaults_on_file": "N",
        "cb_person_delinquencies": 1,
        "cb_person_bankruptcies": 0
    },
    "⚠️ Subprime Borrower (High Risk)": {
        "person_age": 24,
        "person_income": 30000,
        "person_home_ownership": "RENT",
        "person_emp_length": 1,
        "loan_intent": "DEBTCONSOLIDATION",
        "loan_amnt": 19000,
        "loan_int_rate": 21.5,
        "cb_person_cred_hist_length": 2,
        "total_credit_limit": 9000,
        "total_credit_used": 8200,
        "existing_loans_count": 5,
        "previous_loan_defaults_on_file": "Y",
        "cb_person_delinquencies": 3,
        "cb_person_bankruptcies": 1
    }
}

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("⚙️ Model Configuration")
    model_choice = st.selectbox(
        "Choose Classification Model:",
        options=list(all_models.keys()),
        index=list(all_models.keys()).index(champion_name) if champion_name in all_models else 0
    )
    st.caption(f"🏆 Champion Model: **{champion_name}**")
    st.markdown("---")
    
    st.subheader("📋 Load Sample Applicant")
    preset_choice = st.selectbox("Choose a scenario:", list(PRESETS.keys()))
    default_vals = PRESETS[preset_choice] if preset_choice and PRESETS[preset_choice] else PRESETS["🌟 Prime Borrower (Low Risk)"]
    
    st.markdown("---")
    st.subheader("👤 Applicant Profile Inputs")
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        age = st.number_input("Age (Years)", min_value=18, max_value=85, value=default_vals["person_age"])
        emp_length = st.number_input("Employment (Years)", min_value=0, max_value=45, value=default_vals["person_emp_length"])
    with col_sb2:
        income = st.number_input("Annual Income ($)", min_value=5000, max_value=500000, value=default_vals["person_income"], step=5000)
        home_ownership = st.selectbox(
            "Home Ownership",
            ['RENT', 'MORTGAGE', 'OWN', 'OTHER'],
            index=['RENT', 'MORTGAGE', 'OWN', 'OTHER'].index(default_vals["person_home_ownership"])
        )
        
    st.markdown("---")
    st.subheader("💰 Requested Loan Details")
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=100000, value=default_vals["loan_amnt"], step=1000)
    col_sb3, col_sb4 = st.columns(2)
    with col_sb3:
        loan_intent = st.selectbox(
            "Loan Purpose",
            ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
            index=['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'].index(default_vals["loan_intent"])
        )
    with col_sb4:
        interest_rate = st.number_input("Interest Rate (%)", min_value=3.0, max_value=35.0, value=float(default_vals["loan_int_rate"]), step=0.5)
        
    st.markdown("---")
    st.subheader("📜 Credit History & Liabilities")
    col_sb5, col_sb6 = st.columns(2)
    with col_sb5:
        cred_hist = st.number_input("Credit History (Years)", min_value=1, max_value=45, value=default_vals["cb_person_cred_hist_length"])
        total_limit = st.number_input("Total Credit Limit ($)", min_value=500, max_value=300000, value=default_vals["total_credit_limit"], step=2000)
        delinquencies = st.number_input("Past Delinquencies (30+ days)", min_value=0, max_value=10, value=default_vals["cb_person_delinquencies"])
    with col_sb6:
        existing_loans = st.number_input("Existing Open Accounts", min_value=1, max_value=25, value=default_vals["existing_loans_count"])
        total_used = st.number_input("Credit Used / Balance ($)", min_value=0, max_value=300000, value=default_vals["total_credit_used"], step=1000)
        bankruptcies = st.number_input("Bankruptcies on Record", min_value=0, max_value=5, value=default_vals["cb_person_bankruptcies"])
        
    prev_default = "Y" if delinquencies > 0 else "N"

# ----------------- MAIN CONTENT -----------------
st.title("💳 Intelligent Credit Scoring & Risk Underwriting System")
st.markdown("Predict an individual's creditworthiness and probability of loan default using supervised classification algorithms.")

# Prepare Applicant Input DataFrame
applicant_raw = pd.DataFrame([{
    "person_age": age,
    "person_income": income,
    "person_home_ownership": home_ownership,
    "person_emp_length": emp_length,
    "loan_intent": loan_intent,
    "loan_amnt": loan_amnt,
    "loan_int_rate": interest_rate,
    "cb_person_cred_hist_length": cred_hist,
    "total_credit_limit": total_limit,
    "total_credit_used": total_used,
    "existing_loans_count": existing_loans,
    "previous_loan_defaults_on_file": prev_default,
    "cb_person_delinquencies": delinquencies,
    "cb_person_bankruptcies": bankruptcies
}])

# Engineer features
df_eng = engineer_financial_features(applicant_raw)
cat_cols, num_cols = get_column_definitions()
X_input = df_eng[cat_cols + num_cols]

# Transform with pipeline
X_proc = preprocessor.transform(X_input)

# Model Prediction
active_model = all_models[model_choice]
if hasattr(active_model, '__class__') and active_model.__class__.__name__ == 'LogisticRegression':
    if not hasattr(active_model, 'multi_class'):
        active_model.multi_class = 'auto'

prob_default = float(active_model.predict_proba(X_proc)[0, 1])
credit_score = scorecard.pd_to_score(prob_default)
tier_info = scorecard.get_risk_tier(credit_score)

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {tier_info['color']};">
        <span style="color: #666; font-size: 0.85rem; font-weight: 600;">ESTIMATED CREDIT SCORE</span>
        <p class="score-badge" style="color: {tier_info['color']};">{credit_score} <span style="font-size: 1rem; color: #777;">/ 850</span></p>
        <span class="tier-badge" style="color: {tier_info['color']};">{tier_info['tier']} Risk</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    prob_pct = prob_default * 100
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #6c757d;">
        <span style="color: #666; font-size: 0.85rem; font-weight: 600;">DEFAULT PROBABILITY (PD)</span>
        <p class="score-badge" style="color: {'#dc3545' if prob_pct > 35 else '#28a745'};">{prob_pct:.1f}%</p>
        <span style="color: #555; font-size: 0.9rem;">Likelihood of Missed Payments</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    lti = df_eng['loan_to_income_ratio'].iloc[0] * 100
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {'#dc3545' if lti > 40 else '#1e88e5'};">
        <span style="color: #666; font-size: 0.85rem; font-weight: 600;">LOAN-TO-INCOME (LTI)</span>
        <p class="score-badge" style="color: {'#dc3545' if lti > 40 else '#1e88e5'};">{lti:.1f}%</p>
        <span style="color: #555; font-size: 0.9rem;">{f'⚠️ Exceeds 40%' if lti > 40 else 'Healthy Ratio'}</span>
    </div>
    """, unsafe_allow_html=True)

with col4:
    util = df_eng['credit_utilization_ratio'].iloc[0] * 100
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {'#dc3545' if util > 65 else '#ffc107' if util > 30 else '#28a745'};">
        <span style="color: #666; font-size: 0.85rem; font-weight: 600;">CREDIT UTILIZATION</span>
        <p class="score-badge" style="color: {'#dc3545' if util > 65 else '#ffc107' if util > 30 else '#28a745'};">{util:.1f}%</p>
        <span style="color: #555; font-size: 0.9rem;">{f'High Burden (>65%)' if util > 65 else 'Standard'}</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Underwriting Decision Banner
decision_color = tier_info['color']
st.markdown(f"""
<div style="background-color: {decision_color}18; border-left: 6px solid {decision_color}; padding: 14px 20px; border-radius: 8px; margin-bottom: 25px;">
    <h3 style="margin: 0; color: {decision_color};">Underwriting Decision: {tier_info['decision']}</h3>
    <p style="margin: 6px 0 0 0; font-size: 1rem; color: #2d3748;"><b>Analysis:</b> {tier_info['explanation']}</p>
</div>
""", unsafe_allow_html=True)

# ----------------- TABS -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Financial Diagnostics",
    "📈 Model Benchmark & Metrics",
    "🧠 Feature Importance",
    "📖 Beginner Guide & Explanation"
])

# TAB 1: Financial Diagnostics
with tab1:
    st.subheader("Key Financial Health Indicators")
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        dti = df_eng['debt_to_income_ratio'].iloc[0]
        st.write(f"**Debt-To-Income (DTI) Ratio: {dti * 100:.1f}%**")
        st.progress(min(1.0, float(dti)))
        st.caption("Standard underwriting rule: DTI below 36% is optimal; above 43% poses debt burden.")
        
        util_ratio = df_eng['credit_utilization_ratio'].iloc[0]
        st.write(f"**Credit Card / Line Utilization: {util_ratio * 100:.1f}%**")
        st.progress(min(1.0, float(util_ratio)))
        st.caption("FICO recommends keeping revolving credit utilization below 30%.")

    with d_col2:
        st.write(f"**Emergency Credit Buffer:** `${df_eng['credit_available'].iloc[0]:,.2f}`")
        st.write(f"**Active Risk Flags:** `{df_eng['risk_flags_count'].iloc[0]}` out of 4 indicators triggered")
        st.write(f"**Credit History Proportion:** `{df_eng['cred_hist_to_age_ratio'].iloc[0]*100:.1f}%` of adult life")
        
        if df_eng['risk_flags_count'].iloc[0] == 0:
            st.success("✅ Clean financial record! No critical credit stress signals identified.")
        else:
            st.warning(f"⚠️ {df_eng['risk_flags_count'].iloc[0]} warning signals detected. Review debt levels and past payments.")

# TAB 2: Model Benchmark & Metrics
with tab2:
    st.subheader("Classification Models Performance Benchmark")
    st.markdown("All models were trained on 80% of data and evaluated strictly on the 20% held-out test set.")
    
    metrics_records = metadata.get("all_metrics", [])
    if metrics_records:
        metrics_table = pd.DataFrame(metrics_records)
        st.dataframe(metrics_table.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], color='#d4edda'), use_container_width=True)
        
    vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')
    
    v_col1, v_col2 = st.columns(2)
    roc_img = os.path.join(vis_dir, 'roc_curves.png')
    cm_img = os.path.join(vis_dir, 'confusion_matrices.png')
    comp_img = os.path.join(vis_dir, 'model_comparison.png')
    
    with v_col1:
        if os.path.exists(roc_img):
            st.image(roc_img, caption="ROC Curves (Higher Area = Better Discrimination Power)", use_container_width=True)
    with v_col2:
        if os.path.exists(comp_img):
            st.image(comp_img, caption="Side-by-Side Metrics Benchmark", use_container_width=True)
            
    if os.path.exists(cm_img):
        st.image(cm_img, caption="Confusion Matrices across Algorithms", use_container_width=True)

# TAB 3: Feature Importance
with tab3:
    st.subheader("What Drives Credit Worthiness?")
    st.markdown("Feature importance tells us which financial indicators have the biggest influence on the algorithm's predictions.")
    
    fi_img = os.path.join(vis_dir, 'feature_importance.png')
    if os.path.exists(fi_img):
        st.image(fi_img, caption="Top Key Financial Drivers Identified by Random Forest", use_container_width=True)
        
    st.info("""
    💡 **Key Insight for Borrowers:**
    1. **Credit Utilization & Loan-to-Income:** The primary factors in credit scoring. Lowering credit card balances is the quickest way to boost scores.
    2. **Past Delinquencies:** Missing loan or credit card payments heavily inflates default probability.
    3. **Income & Credit History Length:** Consistent tenure and sustained income stability serve as defensive buffers against default.
    """)

# TAB 4: Beginner Guide & Explanation
with tab4:
    st.subheader("📚 Beginner's Guide: Understanding Credit Scoring & Machine Learning")
    
    with st.expander("1. What is Credit Scoring?", expanded=True):
        st.markdown("""
        **Credit Scoring** is a mathematical evaluation method used by financial institutions (banks, NBFCs, fintech lenders)
        to assess how likely an individual is to repay their loan on time.
        
        - **Target Variable (`loan_status`):**
          - `0` = **Non-Default / Creditworthy**: Borrower repays the loan on schedule.
          - `1` = **Default / High Risk**: Borrower fails to pay for 90+ days or charges off.
        """)
        
    with st.expander("2. Why not just rely on 'Accuracy'?"):
        st.markdown("""
        In real banking, defaults are relatively rare (e.g., 5% to 20% of applicants).
        
        If 90% of borrowers never default, a dumb model predicting "Nobody will default" gets **90% Accuracy**,
        yet misses 100% of bad loans!
        
        That is why we evaluate:
        - **Recall (Sensitivity):** Out of all true defaulters, what fraction did the model successfully intercept? (Protects the bank from bad debt).
        - **Precision:** Out of all applicants flagged as defaulters, how many truly were? (Prevents rejecting good customers).
        - **ROC-AUC (Area Under ROC Curve):** Measures overall capability to distinguish good vs bad borrowers across all cutoffs.
        """)
        
    with st.expander("3. How do the 3 Algorithms Compare?"):
        st.markdown("""
        | Algorithm | Strengths | Typical Role in Banking |
        | :--- | :--- | :--- |
        | **Logistic Regression** | Linear, fast, transparent coefficients. Easily maps into regulatory scorecards. | Industry standard for traditional banking compliance. |
        | **Decision Tree** | Intuitive IF-THEN rules, captures non-linear relationships. | Great for explaining decisions to loan officers. |
        | **Random Forest** | Ensemble of many trees; captures complex patterns, resilient to outliers. | State-of-the-art benchmark for modern fintech underwriting. |
        """)
        
    with st.expander("4. How does Probability turn into a 300 - 850 Credit Score?"):
        st.markdown("""
        We use the standard industry **Points to Double the Odds (PDO)** formula:
        $$\\text{Odds} = \\frac{1 - P(\\text{Default})}{P(\\text{Default})}$$
        $$\\text{Score} = \\text{Offset} + \\text{Factor} \\times \\ln(\\text{Odds})$$
        
        - If default probability is **low** (e.g. 2%), Odds are high (49:1) $\\rightarrow$ **Score 750+ (Prime)**.
        - If default probability is **high** (e.g. 50%), Odds are 1:1 $\\rightarrow$ **Score ~550 (Subprime)**.
        """)
