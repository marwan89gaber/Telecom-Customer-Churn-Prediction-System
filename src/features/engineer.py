import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("feature_engineer")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 7 business-driven features on top of the cleaned customer DataFrame.
    All features are derived from domain knowledge about telecom churn.
    Input: clean DataFrame from transform.py
    Output: enriched DataFrame ready for modelling
    """
    logger.info("Starting feature engineering")
    df = df.copy()

    # 1. Tenure bucket
    # Business reason: churn risk is not linear with tenure.
    # New customers (0-12m) are highest risk — they haven't committed yet.
    # Customers past 48m are loyalists — very unlikely to leave.
    df["tenure_bucket"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, float("inf")],
    labels=["new", "developing", "established", "loyal"],
)
    logger.info("Feature: tenure_bucket created")

    # 2. Number of services subscribed
    # Business reason: customers with more services have higher switching costs.
    # A customer with phone + internet + TV + streaming is harder to poach
    # than one with just phone.
    service_cols = [
        "phoneservice", "multiplelines", "internetservice",
        "onlinesecurity", "onlinebackup", "deviceprotection",
        "techsupport", "streamingtv", "streamingmovies",
    ]
    # Count columns where the value is not "No" and not "No internet service" etc.
    df["num_services"] = df[service_cols].apply(
        lambda row: sum(
            1 for v in row
            if str(v).strip() not in {"No", "No phone service",
                                       "No internet service", "nan"}
        ),
        axis=1,
    )
    logger.info("Feature: num_services created")

    # 3. Has protective services flag
    # Business reason: customers with tech support OR online security
    # have invested in the relationship — churn rate is measurably lower.
    df["has_support_services"] = (
        (df["techsupport"] == "Yes") | (df["onlinesecurity"] == "Yes")
    ).astype(int)
    logger.info("Feature: has_support_services created")

    # 4. Charge per tenure month (average monthly spend rate)
    # Business reason: isolates whether a customer is paying proportionally
    # more than peers at the same tenure — a price dissatisfaction signal.
    df["charge_per_tenure"] = df["monthlycharges"] / (df["tenure"] + 1)
    logger.info("Feature: charge_per_tenure created")

    # 5. Monthly spend delta
    # Business reason: compares the current monthly bill against the
    # customer's historical average spend. A sudden increase may signal
    # pricing dissatisfaction and increased churn risk.
    df["monthly_spend_delta"] = (
        df["monthlycharges"]
        - (df["totalcharges"] / (df["tenure"] + 1))
    )
    logger.info("Feature: monthly_spend_delta created")

    # 6. Is new customer
    # Business reason: tenure <= 6 months — the critical onboarding window.
    # Retention interventions in this window have the highest ROI.
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    logger.info("Feature: is_new_customer created")

    # 7. Contract risk score
    # Business reason: encodes contract type as an ordinal risk signal.
    # Month-to-month = 2 (highest risk), one year = 1, two year = 0 (lowest).
    contract_risk = {
        "Month-to-month": 2,
        "One year":        1,
        "Two year":        0,
    }
    df["contract_risk_score"] = df["contract"].map(contract_risk).fillna(1)
    logger.info("Feature: contract_risk_score created")

    # 8. Binary churn target for ML
    # Converts Yes/No churn labels into 1/0 for model training
    df["churn_binary"] = (df["churn"] == "Yes").astype(int)
    logger.info("Feature: churn_binary created")

    logger.info(f"Feature engineering complete — {len(df.columns)} total columns")
    return df


def get_feature_columns() -> list:
    """Returns the list of columns used for ML training in Phase 4."""
    return [
        # Original numeric
        "tenure", "monthlycharges", "totalcharges", "seniorcitizen",
        # Engineered numeric
        "num_services", "has_support_services", "charge_per_tenure", "monthly_spend_delta",
        "is_high_value", "is_new_customer", "contract_risk_score",
        # Categoricals (will be encoded in Phase 4)
        "gender", "partner", "dependents", "phoneservice", "multiplelines", 
        "internetservice", "onlinesecurity", "onlinebackup", "deviceprotection", 
        "techsupport", "streamingtv", "streamingmovies",
        "paperlessbilling", "paymentmethod", "tenure_bucket",
    ]