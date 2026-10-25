"""
predict.py
----------
Command-line inference tool to evaluate an applicant's creditworthiness
and compute their credit score using the trained model and scorecard.
"""

import os
import sys
import joblib
import pandas as pd

# Add workspace to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_engineering import engineer_financial_features, get_column_definitions
from src.scorecard import CreditScorecard


def load_artifacts(models_dir: str = 'models'):
    model_path = os.path.join(models_dir, 'best_model.pkl')
    preproc_path = os.path.join(models_dir, 'preprocessor.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(preproc_path):
        raise FileNotFoundError("Trained model or preprocessor not found. Please run 'python main.py' first.")
        
    model = joblib.load(model_path)
    if hasattr(model, '__class__') and model.__class__.__name__ == 'LogisticRegression':
        if not hasattr(model, 'multi_class'):
            model.multi_class = 'auto'
            
    preprocessor = joblib.load(preproc_path)
    return model, preprocessor


def evaluate_applicant(applicant_data: dict, model=None, preprocessor=None):
    """
    Evaluates a single applicant's financial profile.
    
    Returns:
    --------
    dict containing:
        - Probability of Default (PD)
        - Credit Score (300 - 850)
        - Underwriting Decision
        - Risk Tier & Color
        - Key Financial Ratios
    """
    if model is None or preprocessor is None:
        model, preprocessor = load_artifacts()
        
    df = pd.DataFrame([applicant_data])
    
    # Run Feature Engineering
    df_engineered = engineer_financial_features(df)
    
    cat_cols, num_cols = get_column_definitions()
    X = df_engineered[cat_cols + num_cols]
    
    # Preprocess
    X_proc = preprocessor.transform(X)
    
    # Predict default probability (class 1)
    prob_default = float(model.predict_proba(X_proc)[0, 1])
    is_defaulter = int(model.predict(X_proc)[0])
    
    # Scorecard calculation
    scorecard = CreditScorecard()
    score = scorecard.pd_to_score(prob_default)
    tier_info = scorecard.get_risk_tier(score)
    
    return {
        "applicant_name": applicant_data.get("name", "Applicant"),
        "credit_score": score,
        "probability_of_default": round(prob_default * 100, 2),
        "risk_tier": tier_info["tier"],
        "decision": tier_info["decision"],
        "explanation": tier_info["explanation"],
        "ratios": {
            "Loan-to-Income": f"{df_engineered['loan_to_income_ratio'].iloc[0]*100:.1f}%",
            "Credit Utilization": f"{df_engineered['credit_utilization_ratio'].iloc[0]*100:.1f}%",
            "Debt-to-Income (DTI)": f"{df_engineered['debt_to_income_ratio'].iloc[0]*100:.1f}%",
            "Available Credit": f"${df_engineered['credit_available'].iloc[0]:,.2f}",
            "Risk Flags Count": int(df_engineered['risk_flags_count'].iloc[0])
        }
    }


def main():
    print("="*65)
    print("      CREDIT SCORING APPLICANT EVALUATION (CLI DEMO)")
    print("="*65)
    
    model, preprocessor = load_artifacts()
    
    # Sample 1: Low Risk / Prime Applicant
    applicant_prime = {
        "name": "Sarah Jenkins (Prime Applicant)",
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
    }
    
    # Sample 2: High Risk / Subprime Applicant
    applicant_subprime = {
        "name": "Michael Reed (Subprime Applicant)",
        "person_age": 24,
        "person_income": 32000,
        "person_home_ownership": "RENT",
        "person_emp_length": 1,
        "loan_intent": "DEBTCONSOLIDATION",
        "loan_amnt": 18000,
        "loan_int_rate": 21.0,
        "cb_person_cred_hist_length": 2,
        "total_credit_limit": 10000,
        "total_credit_used": 8900,
        "existing_loans_count": 5,
        "previous_loan_defaults_on_file": "Y",
        "cb_person_delinquencies": 3,
        "cb_person_bankruptcies": 1
    }
    
    for applicant in [applicant_prime, applicant_subprime]:
        res = evaluate_applicant(applicant, model, preprocessor)
        print(f"\nProfile: {res['applicant_name']}")
        print("-" * 50)
        print(f"Credit Score:            {res['credit_score']} / 850")
        print(f"Risk Tier:               {res['risk_tier']}")
        print(f"Default Probability:     {res['probability_of_default']}%")
        print(f"Decision:                {res['decision']}")
        print(f"Underwriting Note:       {res['explanation']}")
        print("Financial Metrics:")
        for k, v in res['ratios'].items():
            print(f"  - {k:22}: {v}")
    
    print("\n" + "="*65)


if __name__ == '__main__':
    main()
