"""
generate_data.py
----------------
Generates a realistic credit scoring dataset mirroring real-world banking
and credit bureau data (such as FICO / Equifax / Experian).

Features:
- Demographics: Age, Employment Length, Home Ownership
- Financials: Annual Income, Loan Amount, Interest Rate, Loan Intent
- Credit History: Credit Line Age, Total Credit Limit, Credit Used,
                 Existing Loans Count, Past Delinquencies, Bankruptcies
- Target: loan_default (0 = Non-Defaulter / Creditworthy, 1 = Defaulter / High Risk)
"""

import os
import numpy as np
import pandas as pd

def generate_credit_dataset(num_samples: int = 7500, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    # 1. Age (21 to 70)
    age = np.random.randint(21, 70, size=num_samples)
    
    # 2. Employment Years (bounded by age - 18)
    max_emp = np.maximum(0, age - 21)
    employment_years = np.array([np.random.randint(0, max(1, m + 1)) for m in max_emp])
    
    # 3. Annual Income (Log-normal distribution, realistic spread: $25k to $220k)
    # mean log income ~ 11.0 (~$60,000)
    income_raw = np.random.lognormal(mean=11.0, sigma=0.55, size=num_samples)
    annual_income = np.clip(np.round(income_raw, -2), 18000, 250000)
    
    # 4. Home Ownership
    home_ownership = np.random.choice(
        ['RENT', 'MORTGAGE', 'OWN', 'OTHER'],
        size=num_samples,
        p=[0.48, 0.40, 0.10, 0.02]
    )
    
    # 5. Loan Intent
    loan_intent = np.random.choice(
        ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
        size=num_samples,
        p=[0.20, 0.18, 0.15, 0.17, 0.12, 0.18]
    )
    
    # 6. Loan Amount ($1,000 to $45,000, correlated moderately with income)
    loan_amount = np.clip(
        np.round(annual_income * np.random.uniform(0.08, 0.45) + np.random.normal(0, 2500), -2),
        1500, 50000
    )
    
    # 7. Interest Rate (8% to 23%)
    interest_rate = np.round(np.random.uniform(7.5, 22.5, size=num_samples), 2)
    
    # 8. Credit History Length (years, correlated with age)
    max_cred_hist = np.maximum(1, age - 20)
    credit_history_years = np.array([np.random.randint(1, max(2, min(30, m + 1))) for m in max_cred_hist])
    
    # 9. Total Credit Limit ($3,000 to $120,000)
    total_credit_limit = np.clip(
        np.round(annual_income * np.random.uniform(0.3, 0.9) + np.random.normal(0, 4000), -2),
        3000, 150000
    )
    
    # 10. Total Credit Used (Balance outstanding)
    utilization_factor = np.random.beta(a=2, b=3, size=num_samples) # spreads across 10% to 90%
    total_credit_used = np.round(total_credit_limit * utilization_factor, 2)
    
    # 11. Existing Loans Count (1 to 10)
    existing_loans_count = np.random.poisson(lam=2.8, size=num_samples)
    existing_loans_count = np.clip(existing_loans_count, 1, 12)
    
    # 12. Past Delinquencies (0 to 6, heavily skewed towards 0)
    delinq_probs = [0.68, 0.18, 0.08, 0.04, 0.015, 0.005]
    past_delinquencies = np.random.choice(len(delinq_probs), size=num_samples, p=delinq_probs)
    
    # 13. Historical Bankruptcies
    bankruptcies = np.random.choice([0, 1, 2], size=num_samples, p=[0.92, 0.07, 0.01])
    
    # -------------------------------------------------------------
    # Target Variable Simulation: Probability of Default (Risk Score)
    # -------------------------------------------------------------
    # Financial logic for default risk:
    # + Higher Credit Utilization -> Higher Risk
    # + Higher Loan to Income -> Higher Risk
    # + Higher Interest Rate -> Higher Risk
    # + Higher Past Delinquencies & Bankruptcies -> Higher Risk
    # - Higher Income -> Lower Risk
    # - Longer Employment & Credit History -> Lower Risk
    # - Homeownership (OWN/MORTGAGE) -> Lower Risk
    
    loan_to_income = loan_amount / annual_income
    credit_utilization = total_credit_used / total_credit_limit
    
    # Financial credit risk logit formula
    logit = (
        -3.2
        + 4.6 * credit_utilization
        + 4.8 * loan_to_income
        + 1.05 * past_delinquencies
        + 1.45 * bankruptcies
        + 0.08 * (interest_rate - 12.0)
        - 0.06 * employment_years
        - 0.07 * credit_history_years
        - 0.60 * (home_ownership == 'OWN').astype(int)
        - 0.30 * (home_ownership == 'MORTGAGE').astype(int)
        + 0.40 * (home_ownership == 'RENT').astype(int)
        + np.random.normal(0, 0.40, size=num_samples) # Realistic noise
    )
    
    prob_default = 1.0 / (1.0 + np.exp(-logit))
    # Binary target: 1 = Defaulter (High Risk), 0 = Repaid / Non-defaulter (Creditworthy)
    loan_default = (np.random.rand(num_samples) < prob_default).astype(int)
    
    # Assemble DataFrame
    df = pd.DataFrame({
        'person_age': age,
        'person_income': annual_income.astype(int),
        'person_home_ownership': home_ownership,
        'person_emp_length': employment_years,
        'loan_intent': loan_intent,
        'loan_amnt': loan_amount.astype(int),
        'loan_int_rate': interest_rate,
        'cb_person_cred_hist_length': credit_history_years,
        'total_credit_limit': total_credit_limit.astype(int),
        'total_credit_used': total_credit_used.astype(int),
        'existing_loans_count': existing_loans_count,
        'previous_loan_defaults_on_file': np.where(past_delinquencies > 0, 'Y', 'N'),
        'cb_person_delinquencies': past_delinquencies,
        'cb_person_bankruptcies': bankruptcies,
        'loan_status': loan_default # Target (1 = Default, 0 = Non-default)
    })
    
    return df

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(data_dir, 'credit_data.csv')
    
    print("Generating synthetic credit scoring dataset...")
    df = generate_credit_dataset(num_samples=7500, random_state=42)
    df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print("\nTarget Class Distribution:")
    print(df['loan_status'].value_counts(normalize=True).round(3))
    print("\nFirst 3 rows:")
    print(df.head(3))
