"""
scorecard.py
------------
Translates machine learning Probability of Default (PD) into an industry-standard
credit score ranging from 300 to 850 (similar to FICO / VantageScore).

Why Credit Scorecards?
- Lenders, loan officers, and customers find numbers like '740' much easier
  to interpret and communicate than a raw probability like '0.048'.
- In credit risk, scores are mapped using the 'Odds of Good vs Bad' and
  'Points to Double the Odds' (PDO) standard methodology:
    Odds = (1 - PD) / PD
    Score = Offset + Factor * ln(Odds)
"""

import numpy as np


class CreditScorecard:
    """
    Standard Credit Bureau Scorecard Converter.
    
    Parameters:
    -----------
    base_score : int (default 660)
        The baseline score corresponding to the base odds.
    base_odds : float (default 10.0)
        Odds of Good:Bad at base score (10 means 10 good loans for every 1 default).
    pdo : int (default 40)
        Points to Double Odds (Score increases by `pdo` points whenever odds double).
    min_score : int (default 300)
        Minimum credit score limit.
    max_score : int (default 850)
        Maximum credit score limit.
    """
    def __init__(self, base_score: int = 660, base_odds: float = 10.0, pdo: int = 40,
                 min_score: int = 300, max_score: int = 850):
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        self.min_score = min_score
        self.max_score = max_score
        
        # Calculate scorecard scaling parameters
        # Factor = PDO / ln(2)
        self.factor = self.pdo / np.log(2.0)
        # Offset = Base_Score - Factor * ln(Base_Odds)
        self.offset = self.base_score - self.factor * np.log(self.base_odds)

    def pd_to_score(self, pd_values: np.ndarray | float) -> np.ndarray | int:
        """
        Converts probability of default (PD) into credit score.
        
        Parameters:
        -----------
        pd_values : float or numpy array of probabilities (0.0 to 1.0)
        
        Returns:
        --------
        credit_scores : int or ndarray of integer scores (300 to 850)
        """
        # Guard against 0 or 1 edge cases
        pd_safe = np.clip(np.array(pd_values), 1e-5, 1.0 - 1e-5)
        
        # Odds of non-defaulting (Good vs Bad)
        odds = (1.0 - pd_safe) / pd_safe
        
        # Scorecard formula: Offset + Factor * ln(Odds)
        raw_score = self.offset + self.factor * np.log(odds)
        
        # Clip to credit score boundaries [300, 850]
        final_score = np.clip(np.round(raw_score), self.min_score, self.max_score).astype(int)
        
        if np.isscalar(pd_values):
            return int(final_score.item())
        return final_score

    def get_risk_tier(self, score: int) -> dict:
        """
        Maps numerical credit score to human-readable risk category
        and standard underwriting lending recommendation.
        """
        if score >= 750:
            return {
                "tier": "Exceptional",
                "color": "#1b8a5a",
                "decision": "APPROVED (Preferred Rates)",
                "explanation": "Applicant has excellent creditworthiness with negligible risk of default."
            }
        elif score >= 700:
            return {
                "tier": "Good",
                "color": "#28a745",
                "decision": "APPROVED (Standard Rates)",
                "explanation": "Applicant demonstrates reliable credit behavior. Low risk of default."
            }
        elif score >= 640:
            return {
                "tier": "Fair",
                "color": "#ffc107",
                "decision": "MANUAL REVIEW / CONDITIONAL APPROVAL",
                "explanation": "Moderate risk. Verification of income and existing liabilities recommended."
            }
        elif score >= 580:
            return {
                "tier": "Poor",
                "color": "#fd7e14",
                "decision": "HIGH RISK (High APR / Collateral Required)",
                "explanation": "Significant credit risk. Higher interest rate or co-signer required."
            }
        else:
            return {
                "tier": "Very Poor",
                "color": "#dc3545",
                "decision": "REJECTED (Excessive Default Risk)",
                "explanation": "High probability of default. Credit application declined."
            }
