"""
feature_engineering.py
----------------------
Handles data preprocessing, financial domain feature engineering,
and transformation pipelines for credit risk prediction.

Key Financial Features Engineered:
1. Loan-to-Income Ratio (LTI):
   - Measures how large the requested loan is compared to annual income.
   - Higher LTI = Greater debt burden on the borrower.

2. Credit Utilization Ratio:
   - Percentage of revolving credit limit currently being used.
   - Industry standard: Utilization > 30% begins to negatively affect credit scores,
     and > 70% indicates high credit stress.

3. Estimated Debt-to-Income Ratio (DTI):
   - Combines estimated monthly loan installment and existing revolving debt service
     divided by gross monthly income. A critical banking underwriting metric.

4. Delinquency-to-Loan Ratio:
   - Normalizes past delinquencies against the number of active credit accounts.

5. Credit History to Adult Age Ratio:
   - Proportion of adult life (age - 18) during which borrower has held credit.

6. Total Available Credit Buffer:
   - Unused credit line available for financial emergencies.

7. High-Risk Indicators Count:
   - Count of severe financial distress flags (high utilization, past defaults, bankruptcy).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def engineer_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives new domain-specific financial risk indicators from raw features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing raw applicant records.
        
    Returns:
    --------
    pd.DataFrame:
        DataFrame with new engineered features.
    """
    df = df.copy()
    
    # Avoid division by zero
    epsilon = 1e-6
    safe_income = np.maximum(df['person_income'], 1000.0)
    monthly_income = safe_income / 12.0
    
    # 1. Loan-to-Income Ratio (LTI)
    df['loan_to_income_ratio'] = df['loan_amnt'] / safe_income
    
    # 2. Credit Utilization Ratio
    safe_credit_limit = np.maximum(df['total_credit_limit'], 100.0)
    df['credit_utilization_ratio'] = np.clip(
        df['total_credit_used'] / safe_credit_limit, 0.0, 1.5
    )
    
    # 3. Estimated Monthly Loan Payment (36-month term assumption)
    # Formula: PMT = P * (r*(1+r)^n) / ((1+r)^n - 1)
    monthly_rate = (df['loan_int_rate'] / 100.0) / 12.0
    term_months = 36
    factor = (1 + monthly_rate) ** term_months
    monthly_pmt = df['loan_amnt'] * (monthly_rate * factor) / (factor - 1 + epsilon)
    
    # Estimated monthly minimum payment on existing credit (~3% of credit used)
    existing_revolving_pmt = df['total_credit_used'] * 0.03
    total_monthly_debt = monthly_pmt + existing_revolving_pmt
    
    # Debt-To-Income (DTI)
    df['debt_to_income_ratio'] = np.clip(total_monthly_debt / monthly_income, 0.0, 2.0)
    
    # 4. Delinquency to Loan Ratio
    safe_loans = np.maximum(df['existing_loans_count'], 1)
    df['delinquency_to_loans_ratio'] = df['cb_person_delinquencies'] / safe_loans
    
    # 5. Credit History to Adult Age Ratio
    adult_years = np.maximum(df['person_age'] - 18, 1)
    df['cred_hist_to_age_ratio'] = np.clip(
        df['cb_person_cred_hist_length'] / adult_years, 0.0, 1.0
    )
    
    # 6. Available Credit (Safety buffer)
    df['credit_available'] = np.maximum(0, df['total_credit_limit'] - df['total_credit_used'])
    
    # 7. High-Risk Flags Count (0 to 4)
    # Counts how many warning signs the applicant presents
    risk_flags = (
        (df['credit_utilization_ratio'] > 0.60).astype(int) +
        (df['loan_to_income_ratio'] > 0.30).astype(int) +
        (df['cb_person_delinquencies'] > 0).astype(int) +
        (df['cb_person_bankruptcies'] > 0).astype(int)
    )
    df['risk_flags_count'] = risk_flags
    
    return df


class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for feature engineering.
    Can be seamlessly integrated into a Scikit-Learn Pipeline.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            # If a numpy array or dict is passed, convert to DataFrame
            raise ValueError("Input to FinancialFeatureEngineer must be a pandas DataFrame")
        return engineer_financial_features(X)


def build_preprocessor(categorical_cols: list, numerical_cols: list) -> ColumnTransformer:
    """
    Builds a Scikit-Learn ColumnTransformer that handles:
    - Missing values (Median imputation for numbers, Most frequent for categories)
    - Feature scaling (StandardScaler to standardize mean=0, std=1)
    - Categorical Encoding (OneHotEncoder with handle_unknown='ignore')
    
    Parameters:
    -----------
    categorical_cols : list
        List of categorical column names.
    numerical_cols : list
        List of numerical column names.
        
    Returns:
    --------
    ColumnTransformer:
        Fitted or unfitted scikit-learn preprocessing transformer.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor


def get_column_definitions():
    """
    Returns the expected categorical and numerical column names
    after feature engineering.
    """
    categorical_cols = [
        'person_home_ownership',
        'loan_intent',
        'previous_loan_defaults_on_file'
    ]
    
    numerical_cols = [
        'person_age',
        'person_income',
        'person_emp_length',
        'loan_amnt',
        'loan_int_rate',
        'cb_person_cred_hist_length',
        'total_credit_limit',
        'total_credit_used',
        'existing_loans_count',
        'cb_person_delinquencies',
        'cb_person_bankruptcies',
        'loan_to_income_ratio',
        'credit_utilization_ratio',
        'debt_to_income_ratio',
        'delinquency_to_loans_ratio',
        'cred_hist_to_age_ratio',
        'credit_available',
        'risk_flags_count'
    ]
    
    return categorical_cols, numerical_cols
